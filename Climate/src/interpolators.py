'''
Name: interpolators.py
Desc: Defines common interpolators for climate data. These interpolators are crafted for the CMIP6 data 
    which have longitude and latitude constructed over a roughly evenly spaced grid

'''

import numpy as np

def bilinear(x,y,lon,lat,cmf):
    '''
    bilinear interpolation
    args:
        x: new longitude coordinate
        y: new latitude coordinate 
        lon: array of longitudes from the cmpi6 model
        lat: array of latitudes from the cmpi6 model
        cmf: array of climate model output over the lon x lat grid 
    '''
    # Set dimensions
    nx = lon.shape[0]
    ny = lat.shape[0]

    # Get the boardering points
    try:
        ix1 = np.argwhere(x>=lon)[-1][0]
        ix2 = ix1 + 1
    except:
        ix1 = -1; ix2 = 0
    
    try:
        iy1 = np.argwhere(y>=lat)[-1][0]
        iy2 = iy1 + 1
    except:
        iy1 = -1; iy2 = 0

    # If the new point is contained within the grid then interpolate between neighbors
    # Otherwise, just set equal to boundary
    if iy1 > -1 and ix1 > -1:
        if ix2 < nx and iy2 < ny:
            # Get function values
            f11 = cmf.transpose()[ix1,iy1]
            f12 = cmf.transpose()[ix1,iy2]
            f21 = cmf.transpose()[ix2,iy1]
            f22 = cmf.transpose()[ix2,iy2]

            # Get x and y values in the rectangle
            x1 = lon[ix1]; x2 = lon[ix2]
            y1 = lat[iy1]; y2 = lat[iy2]

            # Get the interpolation
            c = 1/((x2-x1)*(y2-y1))
            s1 = (x2 - x)*(f11*(y2-y) + f12*(y-y1))
            s2 = (x - x1)*(f21*(y2-y) + f22*(y-y1))
            fxy = c*(s1+s2)
        elif ix2 < nx:
            # Linear interpolation in x-direction
            x1 = lon[ix1]; x2 = lon[ix2]
            f1m = cmf.transpose()[ix1,ny-1]
            f2m = cmf.transpose()[ix2,ny-1]
            
            # Interpolation terms
            c = 1/(x2-x1)
            fxy = c*(f1m*(x2-x) + f2m*(x-x1))
        elif iy2 < ny:
            # Linear interpolation in y-direction
            y1 = lat[iy1]; y2 = lon[iy2]
            fm1 = cmf.transpose()[nx-1,iy1]
            fm2 = cmf.transpose()[nx-1,iy2]
            
            # Interpolation terms
            c = 1/(y2-y1)
            fxy = c*(fm1*(y2-y) + fm2*(y-y1))
        else:
            fxy = cmf.transpose()[nx-1,ny-1]
    elif ix1 > -1:
        # y is at or below the boundary
        # Linear interpolation in x-direction
        if ix2 < nx:
            x1 = lon[ix1]; x2 = lon[ix2]
            f1m = cmf.transpose()[ix1,0]
            f2m = cmf.transpose()[ix2,0]
            
            # Interpolation terms
            c = 1/(x2-x1)
            fxy = c*(f1m*(x2-x) + f2m*(x-x1))
        else:
            fxy = cmf.transpose()[nx-1,0]
    elif iy1 >-1:
        # x is at or below the boundary
        # Linear interpolation in y-direction
        if iy2 < ny:
            y1 = lat[iy1]; y2 = lon[iy2]
            fm1 = cmf.transpose()[0,iy1]
            fm2 = cmf.transpose()[0,iy2]
            
            # Interpolation terms
            c = 1/(y2-y1)
            fxy = c*(fm1*(y2-y) + fm2*(y-y1))
        else:
            fxy = cmf.transpose()[0,ny-1]
    else:
        fxy = cmf.transpose()[0,0]
    
    return fxy



def nearest_neighbor(x,y,lon,lat,cmf):
    '''
    Nearest Neighbor interpolation
    args:
        x: new longitude coordinate
        y: new latitude coordinate 
        lon: array of longitudes from the cmpi6 model
        lat: array of latitudes from the cmpi6 model
        cmf: array of climate model output over the lon x lat grid 
    '''
    # Set dimensions
    nx = lon.shape[0]
    ny = lat.shape[0]

    # Get the boardering points
    try:
        ix1 = np.argwhere(x>=lon)[-1][0]
        ix2 = ix1 + 1
    except:
        ix1 = -1; ix2 = 0
    
    try:
        iy1 = np.argwhere(y>=lat)[-1][0]
        iy2 = iy1 + 1
    except:
        iy1 = -1; iy2 = 0

    # If the new point is contained within the grid then interpolate between neighbors
    # Otherwise, just set equal to boundary
    if ix1 >-1 and iy1 >-1:
        if ix2 < nx and iy2 < ny:
            # Get x and y values in the rectangle
            x1 = lon[ix1]; x2 = lon[ix2]
            y1 = lat[iy1]; y2 = lat[iy2]

            # Get euclidean distance
            d11 = np.sqrt((x-x1)**2 + (y-y1)**2)
            d12 = np.sqrt((x-x1)**2 + (y2-y)**2)
            d21 = np.sqrt((x2-x)**2 + (y-y1)**2)
            d22 = np.sqrt((x2-x)**2 + (y2-y)**2)

            inn = np.argmin([d11,d12,d21,d22])
            nn = [(ix1,iy1),(ix1,iy2),(ix2,iy1),(ix2,iy2)][inn]

            # Get function values
            fxy = cmf.transpose()[nn[0],nn[1]]

        elif ix2 < nx:
            # Occurs if y is on or over boundary but y is not
            x1 = lon[ix1]; x2 = lon[ix2]

            d1 = (x-x1)
            d2 = (x2-x)

            inn = np.argmin([d1,d2])
            nn = [(ix1,iy1),(ix2,iy1)][inn]

            # Get function values
            fxy = cmf.transpose()[nn[0],nn[1]]

        elif iy2 < nx:
            # Occurs if x is on or over boundary but y is not
            y1 = lat[iy1]; y2 = lat[iy2]

            d1 = (y-y1)
            d2 = (y2-y)

            inn = np.argmin([d1,d2])
            nn = [(ix1,iy1),(ix1,iy2)][inn]

            # Get function values
            fxy = cmf.transpose()[nn[0],nn[1]]
        else:
            # x and y are at or over the boundary
            fxy = cmf.transpose()[nx-1,ny-1]
    elif ix1 >-1:
        # Occurs if y is on or below the boundary - check if x is over
        if ix2 < nx:
            x1 = lon[ix1]; x2 = lon[ix2]
            d1 = (x-x1)
            d2 = (x2-x)

            inn = np.argmin([d1,d2])
            nn = [(ix1,0),(ix2,0)][inn]

            # Get function values
            fxy = cmf.transpose()[nn[0],nn[1]]
        else:
            fxy = cmf.transpose()[nx-1,0]
    elif iy1 >-1:
        # Occurs if x is on or below the boundary - check if y is over
        if iy2 < ny:
            y1 = lat[iy1]; y2 = lat[iy2]
            d1 = (y-y1)
            d2 = (y2-y)

            inn = np.argmin([d1,d2])
            nn = [(0,iy1),(0,iy2)][inn]

            # Get function values
            fxy = cmf.transpose()[nn[0],nn[1]]
        else:
            fxy = cmf.transpose()[0,ny-1]
    else:
        # x and y are at or over the boundary
        fxy = cmf.transpose()[0,0]
    return fxy


def invdw(x,y,lon,lat,cmf, q = 1, R = None):
    '''
    Nearest Neighbor interpolation
    args:
        x: new longitude coordinate
        y: new latitude coordinate 
        lon: array of longitudes from the cmpi6 model
        lat: array of latitudes from the cmpi6 model
        cmf: array of climate model output over the lon x lat grid 
    '''
    # Set dimensions
    nx = lon.shape[0]
    ny = lat.shape[0]
    
    # Compute distances
    d2lon = (x - lon)**2
    d2lat = (y - lat)**2
    
    dmat = np.sqrt(d2lon.repeat(ny).reshape(nx,ny) + d2lat.repeat(nx).reshape(ny,nx).transpose())

    # Check to see if there is a distance of 0
    d0 = np.where(dmat == 0)

    try:
        if d0[0].size == 0:
            if R is None:
                # Compute weights and get weighted prediction without a radius
                wmat = 1/dmat**q
            else:
                # Compute weights and get weighted prediction with a radius
                Rdmat = R-dmat
                wmat_max = (Rdmat + np.abs(Rdmat))/2 # quick way to do the max function
                wmat = wmat_max/(R*dmat)**2 
            # Now get the interpolations
            wsum = wmat.sum()
            fxy = (cmf.transpose()*wmat/wsum).sum()
        else:
            fxy = cmf[d0[0][0],d0[1][0]]
    except:
        print("Radius is too small and no obs lie within sphere - increase radius!")    
    return fxy

#bilinear(271.3, 31.4, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100])
#nearest_neighbor(271.3, 31.4, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100])
#invdw(271.3, 31.4, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100],q = 2)
#invdw(271.3, 31.4, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100],R=10)

#bilinear(380,-90, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100])
#nearest_neighbor(380,-90, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100])
#invdw(380, -90, np.array(lon_data[0]),np.array(lat_data[0]),tas_data[0][100],q = 10)
