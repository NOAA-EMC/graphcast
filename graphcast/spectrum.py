# Helper functions to compute the spherical harmonic transform, using Jax-on-GPU

def generate_spectral_coefs(lat_deg):
    # For a Cartesian product grid spanning latitude (lat_deg) in degrees, return the weighted 
    # coefficients of the associated Legendre function (by latitude) required to compute the 
    # spherical harmonic transform.  These coefficients "bake in" the quadrature weight, so 
    # their application is just a multiply-sum.  The coefficients are normalized such that the
    # transform of the constant '1' field has S00=1, implicitly making the transform unit
    # 1/cycle^2.
    import jax
    import jax.numpy as jnp
    import numpy as np

    lat_deg = np.asanyarray(lat_deg)

    # Get grid size
    NLat = lat_deg.size
    # NLong = lon_deg.size

    # Convert latitude and longitude to radians
    lat_rad = (jnp.array(lat_deg) * jnp.pi / 180)#.astype(jnp.float32)
    # lon_rad = (jnp.array(lon_deg) * jnp.pi / 180).astype(jnp.float32)

    # Spherical harmonic polynomials are defined in terms of sin(latitude),
    # or the z Cartesian coordinate
    cart_z = jnp.sin(lat_rad)#.astype(jnp.float32)

    # Build per-latitude weighting factor for integration; the spherical harmonics will be
    # essentially orthonormal with respect to this weighting
    
    # Midpoint latitudes, re-adding the poles
    lat_mid = jnp.concatenate((jnp.array([-jnp.pi/2]), 
                              0.5*(lat_rad[1:] + lat_rad[:-1]), 
                              jnp.array([jnp.pi/2])))
    # The integrated weight is a finite difference on sin(lat).  Near the equator this is essentially
    # cosine-weighting for latitude, but it also behaves nicely near the poles
    lat_weights = jnp.sin(lat_mid[1:]) - jnp.sin(lat_mid[:-1])
    # Normalize to have sum 1
    lat_weights = (lat_weights / jnp.sum(lat_weights))

    # Get the associated legendre functions for the spherical harmonics up to order NLat-1
    # Note that this computation is only reasonable on the GPU.  Modifying this code for the
    # CPU needs to use scipy itself, which doesn't have an equivalent function that includes
    # normalization

    #temporarily disable - lcui
    ## Test whether the default device is a CPU array, to raise an exception
    #if (jnp.array([0]).devices().issubset(jax.devices('cpu'))):
    #    raise ValueError('Spherical harmonic coefficient generation currently requires a GPU')
    
    NSpec = NLat-1
    leg_coef = jax.scipy.special.lpmn_values(NSpec,NSpec,cart_z,True)

    # Find the normalization factor required to give these harmonics a norm of 1 when
    # square-integrated
    orthonorm_factor_squared = 1/jnp.sum(leg_coef[0,0,:]**2 * lat_weights)
    orthonorm_factor = orthonorm_factor_squared**0.5

    # Since we're only going to spherical harmonic space and not backwards (i.e. going from coefficeints
    # to fields), we can bake these weights directly into the coefficients, turning integration into a sum
    leg_coef_weighted = (leg_coef[:,:,:]*(orthonorm_factor * lat_weights[None,None,:])).astype(jnp.float32)

    # An earlier version of this code precomputed longitude-based trigonometric factors.  A moment's thought
    # finds that this is just computing a Fourier transform the hard way, and baking that transform into the
    # function instead serves the same role without needing a large secondary array.
    
    # # Precompute the longitude-based trigonometric factors
    # trig_factors = jnp.exp(1j*np.arange(leg_coef.shape[0])[:,None]*lon_rad[None,:]).astype(np.complex64)

    # Return the weight-included coefficients (for the forward transform) as a numpy array, freeing the GPU
    # memory until it's needed.  For best performance, the weights should be returned to GPU memory before
    # being used, either by using jax.put_device or by wrapping sht_eval inside a lambda to include
    # the coefficients via closure (i.e. do_sht = lambda f: sht_eval(leg_coef_weighted,f)).  With the latter
    # approach, Jax seems to push the weights to the GPU impicitly on use.
    leg_coef_weighted_numpy = np.array(leg_coef_weighted)
    return (leg_coef_weighted_numpy)

def sht_eval(weighted_coef,arr):
    '''Evaluate the complete spherical harmonic transform for all zonal and total wavenumbers, over
    an (N≥2)-dimension array, using jax.einsum for broadcasting.  This function should be called within the
    context of jax.jit for optimization.
    Parameters:
        weighted_coef[zonal,total,lat]: Associated legendre functions for all (total,zonal) wavenumbers, by
                                        latitude, with quadrature weights baked in.  The range of wavenumbers
                                        to evaluate is implicitly given by the dimension of this array
        arr[...,lat,lon]: Array to evaluate'''    
    import jax
    import jax.numpy as jnp
    import jax.lax
    # For better numerical stability, explicitly break out a quick approximation of the constant mode.
    # In float32, the modes are only really orthogonal to 8 or so decimal places, but that means that a
    # large constant shift such as temperature in K can drastically affect accuracy in the tails of
    # the spectrum
    means = jnp.mean(arr,axis=(-1,-2))
    arr = arr - means[...,None,None] # Subtract mean from given array
    farr = jnp.fft.rfft(arr,norm='forward',axis=-1) # Compute FFT for zonal information, nonnegative wavenumbers only
    sht = jnp.einsum('ijk,...ki->...ij',weighted_coef,farr) # Evaluate spherical harmonics
    # Now, add back in the removed mean at [...,0,0]; this needs to use dynamic_update_slice
    # because Jax arrays are immutable
    sht = jax.lax.dynamic_update_slice(sht,means.reshape(sht.shape[:-2] + (1,1)) + \
                                           sht[...,0,0][...,None,None],(0,)*sht.ndim)
    return sht

def power_spectral_density(fspec):
    '''Given an input N-dimensional spectrum of [..., zonal, total], compute the power spectral density
    by total wavenumber, summing over zonal wavenumbers.  Note that a factor of 2 is required for all
    zonal wavenumbers > 0 because of the underlying real zonal FFT that discards negative wavenumbers'''
    import jax.numpy as jnp
    return jnp.abs(fspec[...,0,:])**2 + 2*jnp.sum(jnp.abs(fspec[...,1:,:])**2,axis=-2)

def cross_spectral_density(speca,specb):
    '''Given two spherical harmonic spectra speca, specb[...,zonal,total], compute the cross spectral density
    by total wavenumber.  Note that this result is real, with a factor of 2 normalization that comes from the
    real-valued zonal FFT discarding negative wavenumbers'''
    import jax.numpy as jnp
    areal = speca.real
    breal = specb.real
    aimag = speca.imag
    bimag = specb.imag
    return (areal[...,0,:] * breal[...,0,:] + \
            2*jnp.sum(areal[...,1:,:] * breal[...,1:,:] + \
                      aimag[...,1:,:] * bimag[...,1:,:],axis=-2))


### Helper functions to perform the above operations on XArray datasets

def sht_ds(ds_in,sht_forward):
    # Given an input prediction (or analysis) as an XArray dataset and a forward transform function, 
    # compute the spherical harmonic transform and return a new dataset.  Suitable for JIT compilation,
    # but sht_forward should be set as a "static" argument.

    from graphcast import xarray_jax
    from jax import tree_util
    import numpy as np

    exploded = xarray_jax.unwrap_vars(ds_in)
    spec_dict = tree_util.tree_map(sht_forward,exploded)

    # Rebuild the dataset, copying over all non-horizontal-position dimensions but adding total and zonal wavenumber.
    # For a cleaner set of dataset coordinates, also suppress any latitude/longitude coordinate.  Otherwise, XArray
    # is happy to keep a "useless" coordinate set around, which can complicate things later on.

    rewrapped = xarray_jax.Dataset( {v : (ds_in[v].dims[:-2] + ('zonal_wavenumber','total_wavenumber'), spec_dict[v]) for v in spec_dict.keys()},
                                   coords=ds_in[set(ds_in.coords) - {'latitude','longitude','lat','lon'}].coords)
    rewrapped = rewrapped.assign_coords(total_wavenumber = ('total_wavenumber', np.arange(ds_in.lat.size)))
    return rewrapped

def power_spectral_density_ds(spec):
    # Given an XArray dataset of spherical power spectra, compute the power spectrum
    # by total wavenumber
    from graphcast import xarray_jax
    from jax import tree_util

    # The PSD computation function does not work with the fully wrapped dataset, so use unwrap_vars
    # to get a tree structure of (key, array) pairings
    exploded = xarray_jax.unwrap_vars(spec)

    # Compute the power spectrum of this dictionary by mapping over the 'pytree'
    power_spec_dict = tree_util.tree_map(power_spectral_density,exploded)

    # Rebuild the xarray dataset, copying other coordinates from the supplied arrays
    power = xarray_jax.Dataset( {v : (spec[v].dims[:-2] + ('total_wavenumber',), power_spec_dict[v]) for v in power_spec_dict.keys()},
                               coords = spec.coords)
    return power

def cross_spectral_density_ds(pspec, tsoec):
    # Given spherical spectra for a prediction and analysis, compute the cross spectrum 
    #  against the analysis, by total wavenumber
    # Suitable for compilation with jax
    from graphcast import xarray_jax
    from jax import tree_util

    # Get a tree representation of the prediction and analysis spectra, "exploding" the
    # XArray dataset
    pexplode = xarray_jax.unwrap_vars(pspec)
    aexplode = xarray_jax.unwrap_vars(tsoec)

    # Compute the power spectrum and cross spectrum
    cross_spec_dict = tree_util.tree_map(cross_spectral_density,pexplode,aexplode)

    # Rebuild the XArrays for return
    cross = xarray_jax.Dataset( {v : (pspec[v].dims[:-2] + ('total_wavenumber',), cross_spec_dict[v]) for v in cross_spec_dict.keys()},
                               coords = pspec.coords)
    return (cross)
