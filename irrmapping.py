
"""
@author: Shuo Wang, Hugo Huerta @ NERC TUAS

Functions supplemented to Bifacial_radiance to perform cell-level irradiance simulation

Bifacial_radiance needed.
"""
import numpy as np
import pandas as pd
import concurrent.futures
from subprocess import Popen, PIPE



def genCors(xcell, ycell, zcell, xcellgap, ycellgap, numcellsx, numcellsy, 
                  xgap, ygap, numpanels, nMods, nRows, 
                  originx, originy, height, pitch, tilt, azimuth, 
                  frontsurfaceoffset=0.001, backsurfaceoffset=0.001, offset=0, 
                  arrayRowRange=[0], modColRange=None, modRowRange=None, cellColRange=None, cellRowRange=None):
    
    '''
    Generate coordinate file of sampling points
    
    Parameters:
    ------------
    xcell : float      
        Width of each cell (X-direction) in the module
    ycell : float      
        Length of each cell (Y-direction) in the module
    zcell: float       
        Thickness of the cell
    xcellgap : float   
        Spacing between cells in the X-direction.
    ycellgap : float   
        Spacing between cells in the Y-direction.    
    numcellsx : int    
        Number of cells in the X-direction within the module
    numcellsy : int    
        Number of cells in the Y-direction within the module
    
    xgap : float
        Panel space in X direction. Separation between modules in a row.
    ygap : float
        Gap between modules arrayed in the Y-direction if any.    
    numpanels : int
        Number of modules arrayed in the Y-direction in a row of module array. e.g. 1-up or 2-up, etc. 
    nMods : int 
        Number of modules arrayed in the X-direction in a row of module array
    nRows : int 
        Number of rows in the module array
    originx: float
        X-cordinates of the center of module series
    originy: float
        Y-cordinates of the center of module series        
    height : numeric
        Height of the center of module series 
    pitch : numeric
        Separation between rows
    tilt : numeric 
        Valid input ranges -90 to 90 degrees
    azimuth : numeric 
        Azimuth of the module series. Measured in decimal degrees East of North. [0 to 180) possible.

    frontsurfaceoffset: float
        Offest for the sampling point. Default 0.001 to be at the surface of cell. 
        If glass=True, 0.001 to be at the surface of cell, 0.006 to be at the surface of glass
    backsurfaceoffset: float
        Offest for the sampling point. Default 0.001 to be at the surface of cell. 
        If glass=True, 0.001 to be at the surface of cell, 0.006 to be at the surface of glass
    offset: float
        Default offset = 0 for fixed system, offset = scene.module.offsetfromaxis for 1-axis tracking system       
        
    arrayRowRange: range object or list of int
        The row number to be sampled in the module array, starting from 0
    modColRange: range object or list of int
        The range of columns of modules in the array to be sampled, starting from 0
        Default None for all the columns
    modRowRange: range object or list of int
        The range of rows of modules in the array to be sampled, starting from 0
        Default None for all the rows        
    cellColRange: range object or list of int
        The range of columns of cells in the module to be sampled, starting from 0
        Default None for all the columns
    cellRowRange: range object or list of int
        The range of rows of cells in the module to be sampled, starting from 0
        Default None for all the rows    

    
    Returns
    -------
    
    linepts_front: str
        cordinate map of front sampling points for rtrace simulation
    linepts_back: str
        cordinate map of back sampling points for rtrace simulation
    '''

    # to calculate the starting point of sampling: here is the left-lower corner
    dtor = np.pi/180.0
    
    # dimension of cell-level model, not used in the construction of optical model, but necessary to determine the step distance for sampling
    x = xcell * numcellsx + xcellgap * (numcellsx-1)
    y = ycell * numcellsy + ycellgap * (numcellsy-1)
    scenex = x + xgap
    sceney = np.round(y*numpanels + ygap*(numpanels-1), 8)
    
    ## create linepts_front and linepts_back
    linepts_front = ""
    linepts_back = ""
    
    zdir = np.cos((tilt)*dtor)
    ydir = np.sin((tilt)*dtor) * np.cos((azimuth)*dtor)
    xdir = np.sin((tilt)*dtor) * np.sin((azimuth)*dtor)
    orient_front = '%0.3f %0.3f %0.3f' % (-xdir, -ydir, -zdir)
    orient_back = '%0.3f %0.3f %0.3f' % (xdir, ydir, zdir)
    
    for rowWanted in arrayRowRange:
        
        modWanted = 1 # start point from the first module
    
        # distance between the center of the target panel and the center of the scene. the center of the scene is set to be at originx and originy
        x0 = (modWanted-1)*scenex - (scenex*(round(nMods/1.99)*1.0-1))		
        y0 = (rowWanted)*pitch - (pitch*(round(nRows / 1.99)*1.0-1))
        
        x1 = (x0 - (xcell+xcellgap) * (round(numcellsx/1.99)*1.0-1)) * np.cos ((180-azimuth)*dtor) - y0 * np.sin((180-azimuth)*dtor)		# coordinates of the center of the target panel
        y1 = (x0 - (xcell+xcellgap) * (round(numcellsx/1.99)*1.0-1)) * np.sin ((180-azimuth)*dtor) + y0 * np.cos((180-azimuth)*dtor)
        z1 = 0
        
        # Edge of Panel
        x2 = (sceney/2.0 - ycell/2) * np.cos((tilt)*dtor) * np.sin((azimuth)*dtor)			# coordinates of the lower edge of the target panel
        y2 = (sceney/2.0 - ycell/2) * np.cos((tilt)*dtor) * np.cos((azimuth)*dtor)
        z2 = -(sceney/2.0 - ycell/2) * np.sin(tilt*dtor)
        
        #offset = 0 # 0 for fixed system
        modulez = zcell
        
        frontsurfaceoffset = frontsurfaceoffset
        backsurfaceoffset = backsurfaceoffset
        # Axis of rotation Offset (if offset is not 0) for the front of the module
        x3 = (offset + modulez + frontsurfaceoffset) * np.sin(tilt*dtor) * np.sin((azimuth)*dtor)		# correction of Z direction
        y3 = (offset + modulez + frontsurfaceoffset) * np.sin(tilt*dtor) * np.cos((azimuth)*dtor)
        z3 = (offset + modulez + frontsurfaceoffset) * np.cos(tilt*dtor)
        
        # Axis of rotation Offset, for the back of the module 
        x4 = (offset - backsurfaceoffset) * np.sin(tilt*dtor) * np.sin((azimuth)*dtor)
        y4 = (offset - backsurfaceoffset) * np.sin(tilt*dtor) * np.cos((azimuth)*dtor)
        z4 = (offset - backsurfaceoffset) * np.cos(tilt*dtor)
        
        xstart_front0 = x1 + x2 + x3 + originx
        xstart_back0 = x1 + x2 + x4 + originx
        
        ystart_front0 = y1 + y2 + y3 + originy
        ystart_back0 = y1 + y2 + y4 + originy
        
        zstart_front0 = height + z1 + z2 + z3
        zstart_back0 = height + z1 + z2 + z4
      
        
        ## to calculate the step distance
      
        # shift of coordinates for distance between cells along y-direction of panel
        dX = (ycell + ycellgap) * np.cos(tilt*dtor) * np.sin((azimuth - 180)*dtor)
        dY = (ycell + ycellgap) * np.cos(tilt*dtor) * np.cos((azimuth - 180)*dtor)
        dZ = (ycell + ycellgap) * np.sin(tilt*dtor)
        
        # shift of coordinates for for distance between cells along x-direction of panel
        sx_dX = (xcell + xcellgap) * np.cos((azimuth - 180)*dtor)
        sx_dY = - (xcell + xcellgap) * np.sin((azimuth - 180)*dtor)
        
        # shift of coordinates for distance between panels along x-direction of panel
        
        mx_dX = (x + xgap) * np.cos((azimuth - 180)*dtor)
        mx_dY = -(x + xgap) * np.sin((azimuth - 180)*dtor)
           
        # shift of coordinates for distance between panels along y-direction of panel
        
        my_dX = (y+ygap) * np.cos(tilt*dtor) * np.sin((azimuth - 180)*dtor)
        my_dY = (y+ygap) * np.cos(tilt*dtor) * np.cos((azimuth - 180)*dtor)
        my_dZ = (y+ygap) * np.sin(tilt*dtor)    

 
        # calculate coordiants of the sampling points   
        if modColRange is None:        
            modColRange = range(nMods)
            
        if modRowRange is None:   
            modRowRange = range(numpanels)
            
        if cellColRange is None:       
            cellColRange = range(numcellsx)
            
        if cellRowRange is None:       
            cellRowRange = range(numcellsy)  
    
        for mod_column in modColRange:  
    

            for mod_row in modRowRange:
            
                xstart_front = xstart_front0 + mx_dX *mod_column + my_dX * mod_row
                ystart_front = ystart_front0 + mx_dY *mod_column + my_dY * mod_row
                zstart_front = zstart_front0 + my_dZ * mod_row
                
                xstart_back = xstart_back0 + mx_dX *mod_column + + my_dX * mod_row
                ystart_back = ystart_back0 + mx_dY *mod_column + my_dY * mod_row
                zstart_back = zstart_back0 + my_dZ * mod_row
                
                for cell_column in cellColRange:
                    
                    for cell_row in cellRowRange:
                        
                        xpos_front = xstart_front + dX * cell_row + sx_dX * cell_column
                        ypos_front = ystart_front + dY * cell_row + sx_dY * cell_column
                        zpos_front = zstart_front + dZ * cell_row + 0 * cell_column
                        linepts_front = linepts_front + str(xpos_front) + ' ' + str(ypos_front) + ' ' + str(zpos_front) + ' ' + orient_front + " \r"
                
                        xpos_back = xstart_back + dX * cell_row + sx_dX * cell_column
                        ypos_back = ystart_back + dY * cell_row + sx_dY * cell_column
                        zpos_back = zstart_back + dZ * cell_row + 0 * cell_column
                        linepts_back = linepts_back + str(xpos_back) + ' ' + str(ypos_back) + ' ' + str(zpos_back) + ' ' + orient_back + " \r"
           

    return linepts_front, linepts_back



def _popen(cmd, data_in, data_out = PIPE):
    """
    --Borrowed from Bifacial_radiance--
    Helper function subprocess.popen replaces os.system
    - gives better input/output process control
    usage: pass <data_in> to process <cmd> and return results
    based on rgbeimage.py (Thomas Bleicher 2010)
    """
    if type(cmd) == str:
        cmd = str(cmd)                                          # gets rid off unicode oddities
        shell = True
    else:
        shell = False

    #initT_1 = time.time()
    p = Popen(cmd, bufsize = -1, stdin = PIPE, stdout = data_out, stderr = PIPE, shell = shell, text = True) #shell=True required for Linux? quick fix, but may be security concern
    data, err = p.communicate(data_in)
    #endT_1 = time.time()
    #print('\nIrradiance values collected with rtrace function in: ', endT_1 - initT_1, '(s)\n')

    if err:
        if data:
            returntuple = (data, 'message: ' + err)
        else:
            returntuple = (None, 'message: ' + err)
    else:
        if data:
            returntuple = (data, None) 
        else:
            returntuple = (None, None)

    return (returntuple)



def _irrMap(octfile:str, linepts:str, mytitle = None, accuracy = 'low'):
    """
    --Partly borrowed from Bifacial_radiance--
    irradiance plotting using rtrace
    pass in the linepts structure of the view along with a title string
    for the plots.  

    Parameters
    ------------
    octfile : string
        Filename and extension of .oct file
    linepts : 
        Output from genCors
    mytitle : string
        Title to append to results files

    accuracy : string
        Either 'low' (default - faster) or 'high'
        (better for low light)

    Returns
    -------
    out : dictionary
        out.x,y,z  - coordinates of point
        .r,g,b     - r,g,b values in Wm-2
        .Wm2            - equal-weight irradiance
        .mattype        - material intersected
        .title      - title passed in
    """
    
    if mytitle is None:
        mytitle = octfile[:-4]

    if octfile is None:
        print('Analysis aborted. octfile = None' )
        return None

    out = { 'Wm2':[],'x':[], 'y':[], 'z':[], 'r':[], 'g':[],'b':[],'mattype':[], 'title':mytitle }
    
    print ('Full scan in process for: %s' %(mytitle))

    if accuracy == 'low':
        #rtrace optimized for faster scans: (ab2, others 96 is too coarse)           
        cmd = "rtrace -i -ab 2 -aa .1 -ar 256 -ad 2048 -as 256 -h -oovs " + octfile

    elif accuracy == 'high':
        #rtrace ambient values set for 'very accurate':
        cmd = "rtrace -i -ab 5 -aa .08 -ar 512 -ad 2048 -as 512 -h -oovs " + octfile
    elif accuracy =='test':
        # for fast tests
        cmd = "rtrace -i -ab 2 -aa .2 -ar 64 -ad 1024 -as 0 -h -oovs " + octfile

    else:
        print('_irrPlot accuracy options: "low" or "high"')
        return({})
   
    temp_out, err = _popen(cmd, linepts)

    if err is not None:
        if err[0:5] == 'error':
            raise Exception(err[7:])
        else:
            print(err)

    # when file errors occur, temp_out is None, and err message is printed.
    if temp_out is not None:

        for line in temp_out.splitlines():
            temp = line.split('\t')
            out['x'].append(float(temp[0]))
            out['y'].append(float(temp[1]))
            out['z'].append(float(temp[2]))
            out['r'].append(float(temp[3]))
            out['g'].append(float(temp[4]))
            out['b'].append(float(temp[5]))
            out['mattype'].append(temp[6])
            out['Wm2'].append(sum([float(i) for i in temp[3:6]])/3.0)

    else:
        out = None                 # return empty if error message.

    return(temp_out, out)          # out is the processed file



def concurMap(radianceObj, lineptsFront, lineptsBack, tRange=None, sys='fixed', accuracy='low'):
    
    '''
    Generate Skyfile & Octfiles and run simulations for all the sampling points with concurrant loop
        
    Parameters:
    ------------
    radianceObj : RadianceObj      
        Created in the simulation

    lineptsFront, lineptsBack: str
           
        Coordinates of the sampling points
        
    t_range : list of DatetimeIndex      
        Starting and ending hour of simulation, in format "YYYY-MM-DD HH:MM:SS tz".
        Default as same as radianceObj.metdata.datetime
    
    accuracy: str  
        Calculation accuracy. 'low' or 'hight'
        
    sys: str
        type of system: 'fixed' or '1axis'. Default = 'fixed'
        
    Output:
    ------------
        Simulation results in .csv files
    '''

    def _sim(octfile, output, lineptsFront=lineptsFront, lineptsBack=lineptsBack, accuracy=accuracy):


        out, frontDict =_irrMap(octfile, lineptsFront, output+'_Front', accuracy)
        out, backDict = _irrMap(octfile, lineptsBack, output+'_Back', accuracy) 
        
        #print("\nGetting irradiance maps for ", output)
        frontDF = pd.DataFrame.from_dict(frontDict)      
        backDF = pd.DataFrame.from_dict(backDict)
        DF = frontDF[['x','y','z', 'mattype']]
        DF.insert(3, "rearZ", backDF['z'])
        DF.insert(5, "rearMat", backDF['mattype'])
        DF.insert(6, "Wm2Front", frontDF['Wm2'])
        DF.insert(7, "Wm2Back", backDF['Wm2'])
        DF['Back/FrontRatio'] = backDF[['Wm2']]/frontDF[['Wm2']]
        DF.to_csv('results/' + output +'.csv', index=False, sep=',')  
    
    
    if tRange is None:
        
        tRange = radianceObj.metdata.datetime
    
    
    if sys == 'fixed':          

     
        fileLists = []
        octNames = []
        
        for stime in tRange:
            timest =  str(stime)[:10] + '_' + str(stime)[11:13] + str(stime)[14:16]
            
            # prevent error when the timestamp is not in the metdata range
            if stime not in radianceObj.metdata.datetime:    
                
                print(str(stime)[:16] + ': no effective metdata.')      
                continue
            
            timestamp = radianceObj.metdata.datetime.index(stime) 
                              
            if ( np.isnan(radianceObj.metdata.ghi[timestamp]) or np.isnan(radianceObj.metdata.dhi[timestamp])):
                
                print(str(stime)[:16] + ': irradiance error.')
                continue
            
         
            skyfile = radianceObj.gendaylit(timestamp)
          
            fileLists.append(radianceObj.materialfiles + [skyfile] + radianceObj.radfiles)
            octNames.append(radianceObj.basename + '_' + timest)
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            octFiles = executor.map(radianceObj.makeOct, fileLists, octNames)
    
        octList = []
        for octfile in octFiles:
          
            octList.append(octfile)
        
    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_sim, octList, octNames)
            
    elif sys == '1axis':
        
        fileLists = []
        octNames = []
        
        trackerdict = radianceObj.gendaylit1axis()
        trackerkeys = sorted(radianceObj.trackerdict.keys())
        
        for stime in tRange:
            
            trackerkey = str(stime)[:10] + '_' + str(stime)[11:13] + str(stime)[14:16]
            
            if trackerkey not in trackerkeys:    
                
                print(str(stime)[:16] + ': no effective metdata.')      
                continue
            
            if ( np.isnan(trackerdict[trackerkey]['ghi']) or np.isnan(trackerdict[trackerkey]['dhi']) ):
                
                print(str(stime)[:16] + ': irradiance error.')
                continue
            
            filelist = radianceObj.materialfiles + [trackerdict[trackerkey]['skyfile'], trackerdict[trackerkey]['radfile']]
            octname = radianceObj.basename + '_1axis_%s'%(trackerkey)
            
            fileLists.append(filelist)
            octNames.append(octname)
        
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            octFiles = executor.map(radianceObj.makeOct, fileLists, octNames)

        octList = []
        for octfile in octFiles:
          
            octList.append(octfile)    
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(_sim, octList, octNames, lineptsFront, lineptsBack)   
        
    
    else:
        print('Please choose the type of system: fixed or 1axis')