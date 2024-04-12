from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.lab.visualization import InteractivePlot, Plot

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.core.problem import OnTheFlyFloatProblem
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

import os
import math
import numpy
import random
import arcpy
import dill
import arcpy
import numpy as np

n_D = 2
Roadwidth =12                       #in meters
roadaccuracy =50                     #in meters
StudyAreaXRange =[-9176837.0,-9170875.0]   #in meters
StudyAreaYRange =[3294621.7,3301272.0]     #in meters

Radius_min = 560
lengthofspiral =0                    #in meters
Alpha0 = 0
Alpha1 = 400
Alpha2 = 1.2

#Unit Costs
All_Intersections = ['VZ_NW13','VZ_NW12','Wetland','Forbidden_Area']
CR = ['VZ_NW13','VZ_NW12']
UnitLocationDependentCost={'Wetland': 0,'Forbidden_Area':2000}
RoadCostperMeter = 600
roadCostpermile = 4363836.78                         #in dolars/mile
arcpyworkspace =r'G:\Thesis\Revised_Article\Case7.gdb'

#Varaiables Depending on Input:

arcpy.env.workspace = arcpyworkspace
srs=arcpy.Describe("StudyArea").spatialReference
arcpy.env.outputCoordinateSystem=srs
arcpy.env.outputZFlag = "DISABLED"

startpoint = list(arcpy.da.FeatureClassToNumPyArray('Start',['SHAPE@X','SHAPE@Y']).tolist()[0])
endpoint = list(arcpy.da.FeatureClassToNumPyArray('End',['SHAPE@X','SHAPE@Y']).tolist()[0])

fcs = arcpy.ListFeatureClasses()
arcpy.conversion.FeatureClassToGeodatabase(fcs, r'in_memory')

def PIX (ABIndi,i): #Return X Coordinate of ith PI in the Individual (road path) (meters)
    if i == -1:
        return round(float(startpoint[0]),2)
    elif i == len(ABIndi):
        return round(float(endpoint[0]),2)
    else:
        return round(float(ABIndi[i][0]),2)
    
def PIY (ABIndi,i): #Return Y Coordinate of ith PI in the Individual (road path) (meters)
    if i == -1:
        return round(float(startpoint[1]),2)
    elif i == len(ABIndi):
        return round(float(endpoint[1]),2)
    else:
        return round(float(ABIndi[i][1]),2)
    
def ArraytoList (ABIndi):
    LBIndi=[]
    for i in range(len(ABIndi)):
        for j in range(n_D):
            LBIndi.append(ABIndi[i][j])
    return LBIndi

def ccw (A,B,C):
    return (C[1]-A[1])*(B[0]-A[0])>(B[1]-A[1])*(C[0]-A[0])

def intersect (seg1, seg2): #Return True if segment A-B intersects segment C-D
    (A,B),(C,D) = seg1, seg2
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def linesegments (ABIndi): #Return a list of all segments which start from (i-1)th PI and ends on ith PI
    return [[[PIX(ABIndi,i-1),PIY(ABIndi,i-1)],[PIX(ABIndi,i),PIY(ABIndi,i)]] for i in range(len(ABIndi)+1)]

def SelfIntersection (ABIndi):
    line_segments = linesegments(ABIndi)
    intersections=[]
    for i in range(len(line_segments)):
        for j in range(i-1):
            if intersect(line_segments[i],line_segments[j]):
                intersections.append(-1) #SelfIntersected
                intersections.append((i,j))
            else:
                intersections.append(1)
    if all(i==1 for i in intersections):
        return 1
    else:
        return -1

def Handling (ABIndi): #THIS FUNCTION HANDLES INDIVIDUALS INTERSECT THEMSELVES (USED IN FIRST POPULATION)
#    print("handling")
    ind = ABIndi
    r=0
    while SelfIntersection(ind)==-1 and r< 2000:
        
        r=r+1
        line_segments = linesegments(ind)
        for k in range(len(ind)):
            for m in range(k+2,len(ind)+1):
                if intersect(line_segments[k],line_segments[m]):
                    ind[k] , ind[m-1] = ind[m-1], ind[k]
                    line_segments = linesegments(ind)

    return ind

def get_simplified_routes(my_workspace):
    
    arcpy.env.workspace = my_workspace
    srs=arcpy.Describe("StudyArea").spatialReference
    arcpy.env.overwriteOutput = True

    arcpy.env.outputCoordinateSystem=srs
    BufferWidth = Roadwidth/2
    convertingcellsize = Roadwidth/2
    arcpy.Buffer_analysis('Forbidden_Area',r'in_memory\Buffered_Forbidden_Area','%s Meters' %BufferWidth)
    arcpy.AddField_management(r'in_memory\Buffered_Forbidden_Area','Value', 'SHORT')
    arcpy.CalculateField_management(r'in_memory\Buffered_Forbidden_Area','Value','10000','PYTHON_9.3')
    arcpy.Merge_management([r'in_memory\Buffered_Forbidden_Area','StudyArea'],r'in_memory\merged_Forbidden')
    arcpy.PolygonToRaster_conversion (r'in_memory\merged_Forbidden','Value',r'in_memory\Raster_Forbidden_Area',priority_field = 'Value',cellsize=convertingcellsize)

    simple_routes_fc = []
    for i in CR:

        arcpy.Buffer_analysis(i,r'in_memory\Buffered_%s' %i,'%s Meters' %BufferWidth)
        arcpy.AddField_management(r'in_memory\Buffered_%s' %i,'Value', 'SHORT')
        arcpy.CalculateField_management(r'in_memory\Buffered_%s' %i,'Value','10000','PYTHON_9.3')
        arcpy.Merge_management([r'in_memory\Buffered_%s' %i,'StudyArea'],r'in_memory\merged_%s' %i)
        arcpy.PolygonToRaster_conversion (r'in_memory\merged_%s' %i,'Value',r'in_memory\Raster_%s' %i,priority_field = 'Value',cellsize=convertingcellsize)
        outRaster = arcpy.sa.WeightedSum(arcpy.sa.WSTable([[r'in_memory\Raster_Forbidden_Area','Value',1],[r'in_memory\Raster_%s' %i,'Value',1]]))
        outRaster.save(rf'in_memory\merged_{i}_Forbidden_raster')

        outCostDist = arcpy.sa.CostDistance('Start',rf'in_memory\merged_{i}_Forbidden_raster',out_backlink_raster = rf'in_memory\merged_{i}_backlink')
        arcpy.sa.CostPathAsPolyline('End',outCostDist,rf'in_memory\merged_{i}_backlink',rf'{i}_complicatedpolyline')
        arcpy.edit.FlipLine(rf'{i}_complicatedpolyline')
        arcpy.Merge_management([i,'Forbidden_Area'],rf'in_memory\barrier'+i)
        arcpy.cartography.SimplifyLine(rf'{i}_complicatedpolyline',rf'{i}_SimpleRoute','POINT_REMOVE',20000,None,'NO_KEEP',None,rf'in_memory\barrier'+i)
        simple_routes_fc.append(rf'{i}_SimpleRoute')

#    print('max_number',max([max(row[0].pointCount for row in arcpy.da.SearchCursor(i,'SHAPE@')) - 2 for i in simple_routes_fc]))
    return simple_routes_fc, max([max(row[0].pointCount for row in arcpy.da.SearchCursor(i,'SHAPE@')) - 2 for i in simple_routes_fc])

def generate_initial_sols (my_workspace,
                           number_of_variables,
                           number_of_objectives,
                           simple_routes_fc,
                           StudyAreaXRange,
                           StudyAreaYRange,
                           pop_size):
    
    arcpy.env.workspace = my_workspace
    srs=arcpy.Describe("StudyArea").spatialReference
    arcpy.env.outputCoordinateSystem=srs
    pop = []
    lower_bound = int(number_of_variables/2) * [min(StudyAreaXRange), min(StudyAreaYRange)]
    upper_bound = int(number_of_variables/2) * [max(StudyAreaXRange), max(StudyAreaYRange)]
    
    for i in simple_routes_fc:
        number_of_vertices = max(row[0].pointCount for row in arcpy.da.SearchCursor(i,'SHAPE@')) - 2
        
        if n_D*number_of_vertices < number_of_variables:
#            print('number_of_vertices',number_of_vertices)
#            print('number_of_variables',number_of_variables)
            number_of_new_vertices = (number_of_variables//n_D) - number_of_vertices
#            print('number_of_new_vertices:',number_of_new_vertices )
            vertices = arcpy.da.FeatureClassToNumPyArray(i,['OID@','SHAPE@XY'], explode_to_points = True)
            lines = arcpy.da.SearchCursor (i, 'SHAPE@')

            existing_vertices = {}
            
            for line in lines:
                for index in range(len(vertices)):
                    existing_vertices['ex'+str(index)] = [line[0].measureOnLine(arcpy.Point(*vertices[index][1]) , True),arcpy.Point(*vertices[index][1])] 


            dummy_vertices = {}
            lines = arcpy.da.SearchCursor (i, 'SHAPE@')
            
            for line in lines:
                for new_vertex in range(1,number_of_new_vertices+1):

                    perc = new_vertex/(number_of_new_vertices+1)
                    dummy_vertices['dm'+str(new_vertex)] = [perc,line[0].positionAlongLine(perc,True).centroid]
            
            all_vertices = list((existing_vertices|dummy_vertices).items())
            all_vertices.sort(key = lambda x: x[1][0])

            features =[]
            features.append(arcpy.Polyline(arcpy.Array([vert[1][1] for vert in all_vertices]), spatial_reference = srs))
            arcpy.CreateFeatureclass_management(out_path=my_workspace,
                                                out_name=fr'{i}_FixedRoute',
                                                geometry_type="POLYLINE",
                                                has_m= "DISABLED",
                                                has_z= "DISABLED",
                                                spatial_reference=srs)

            arcpy.CopyFeatures_management(features, fr'{i}_FixedRoute')

            new_solution = FloatSolution(lower_bound,upper_bound, number_of_objectives)

            arcpy.FeatureVerticesToPoints_management(fr'{i}_FixedRoute','in_memory\\routepoints'+i)
            table= arcpy.da.FeatureClassToNumPyArray('in_memory\\routepoints'+i,['SHAPE@X','SHAPE@Y'])
            table = table[1:-1]
            
            table = np.array([list(coord) for coord in table])
            table = table.flatten()
            table = table.tolist()
            for num in range(number_of_variables):

                new_solution.variables[num] = table[num]
                
            pop.append(new_solution)
            
        elif n_D*number_of_vertices == number_of_variables:
            
            arcpy.Copy_management(i, fr'{i}_FixedRoute')
            new_solution = FloatSolution(lower_bound,upper_bound, number_of_objectives)

            arcpy.FeatureVerticesToPoints_management(fr'{i}_FixedRoute','in_memory\\routepoints'+i)
            table= arcpy.da.FeatureClassToNumPyArray('in_memory\\routepoints'+i,['SHAPE@X','SHAPE@Y'])
            table = table[1:-1]

            table = np.array([list(coord) for coord in table])
            table = table.flatten()
            table = table.tolist()
            
            for num in range(number_of_variables):
                new_solution.variables[num] = table[num]

            pop.append(new_solution)

        else:

            number_of_divided = int(number_of_vertices /(number_of_variables/n_D))+1
            
            inds = {}
            lines = arcpy.da.SearchCursor(i, 'SHAPE@')

            points = arcpy.da.FeatureClassToNumPyArray(i,['OID@','SHAPE@XY'], explode_to_points = True)
            points = [arcpy.Point(*list(pnt[1])) for pnt in points]
            points_without_ends = points[1:-1]

            for ind in range(number_of_divided):
                inds[ind] = [points[0]]

            for ind in range(number_of_divided):
                inds[ind].extend(points_without_ends[ind*number_of_variables//n_D:(ind+1)*number_of_variables//n_D])

            inds[number_of_divided-1] = [points[0]] + points_without_ends[-number_of_variables//n_D:]
            
            for ind in range(number_of_divided):
                inds[ind].extend([points[-1]])

            for ind in range(number_of_divided):
                features =[]
                features.append(
                    arcpy.Polyline(
                        arcpy.Array([point for point in inds[ind]]), spatial_reference = srs))
                arcpy.CreateFeatureclass_management(out_path=my_workspace,
                                                    out_name=fr'{i}_{ind}_FixedRoute',
                                                    geometry_type="POLYLINE",
                                                    has_m= "DISABLED",
                                                    has_z= "DISABLED",
                                                    spatial_reference=srs)

                arcpy.CopyFeatures_management(features, fr'{i}_{ind}_FixedRoute')
                
                new_solution = FloatSolution(lower_bound,upper_bound, number_of_objectives)

                arcpy.FeatureVerticesToPoints_management(fr'{i}_{ind}_FixedRoute','in_memory\\routepoints'+i+str(ind))
                table= arcpy.da.FeatureClassToNumPyArray('in_memory\\routepoints'+i+str(ind),['SHAPE@X','SHAPE@Y'])
                table = table[1:-1]
                table = table.flatten()
                table = table.tolist()
                
                for num in range(number_of_variables):
                    new_solution.variables[num] = table[num]

                pop.append(new_solution)
    pop_to_cad = []
    while len(pop)<pop_size:
        
        new_solution = FloatSolution(lower_bound,upper_bound, number_of_objectives)

        ABSolu = [[random.uniform(min(StudyAreaXRange),max(StudyAreaXRange)),
                   random.uniform(min(StudyAreaYRange),max(StudyAreaYRange))] for _ in range(number_of_variables//n_D)]

        ABSolu = Handling(ABSolu)

        pop_to_cad.append(ABSolu)
        LBSolu = np.array(ABSolu)
        LBSolu = LBSolu.flatten()
        
        for num in range(number_of_variables):
            new_solution.variables[num] = LBSolu[num]

        pop.append(new_solution)

    with open('init_solutions', 'wb') as f1:
        dill.dump(pop, f1)
    with open('init_solutions_to_cad', 'wb') as f2:
        dill.dump(pop_to_cad, f2)

##    for sol_ind in range(len(pop)):
##        sol = pop[sol_ind]
##        for i in range(0,len(sol.variables)-2,2):
##            if sol.variables[i] == sol.variables[i+2] and sol.variables[i+1] == sol.variables[i+3]:
##                print(sol.variables)
##
##    for sol_ind in range(len(pop)):
##        sol = pop[sol_ind]
##        for i in range(0,len(sol.variables)):
##            if i in [-9176837.0,-9170875.0]+[3294621.7,3301272.0]:
##                print (sol.variables)
    
    print('initial generation finished')    
    return pop
        
  
class Alignment_Problem(FloatProblem):

    n_D = 2
    Roadwidth =12                       #in meters
    roadaccuracy =50                     #in meters
    StudyAreaXRange =[-9176837.0,-9170875.0]   #in meters
    StudyAreaYRange =[3294621.7,3301272.0]     #in meters

    Radius_min = 560
    lengthofspiral =0                    #in meters
    Alpha0 = 500000
    Alpha1 = 2000
    Alpha2 = 1.2
    
    #Unit Costs
    All_Intersections = ['VZ_NW13','VZ_NW12','Wetland','Forbidden_Area']
    CR = ['VZ_NW13','VZ_NW12']
    UnitLocationDependentCost={'Wetland': 0,'Forbidden_Area':2000}
    RoadCostperMeter = 600
    roadCostpermile = 4363836.78                         #in dolars/mile
#    arcpyworkspace =r'G:\Thesis\Revised_Article\Case7.gdb'
    arcpyworkspace =r'in_memory'
    #Varaiables Depending on Input:

    arcpy.env.workspace = arcpyworkspace
    srs=arcpy.Describe("StudyArea").spatialReference
    arcpy.env.outputCoordinateSystem=srs
    arcpy.env.outputZFlag = "DISABLED"

    startpoint = list(arcpy.da.FeatureClassToNumPyArray('Start',['SHAPE@X','SHAPE@Y']).tolist()[0])
    endpoint = list(arcpy.da.FeatureClassToNumPyArray('End',['SHAPE@X','SHAPE@Y']).tolist()[0])

    simplified_routes, max_pointcount = get_simplified_routes(arcpyworkspace)
    
    def __init__(self, number_of_variables: int, number_of_objectives : int, impose_number_of_IPs : bool, pop_size: int):
        super(Alignment_Problem, self).__init__()
        
        if not impose_number_of_IPs:
            number_of_variables = self.n_D * self.max_pointcount

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ["SV", "Cost"]

        self.lower_bound = int(number_of_variables/2) * [min(self.StudyAreaXRange), min(self.StudyAreaYRange)]
        self.upper_bound = int(number_of_variables/2) * [max(self.StudyAreaXRange), max(self.StudyAreaYRange)]
        self.impose_number_of_IPs = impose_number_of_IPs
        self.pop_size = pop_size
        self.initial_solutions = generate_initial_sols (arcpyworkspace,
                                                        number_of_variables,
                                                        number_of_objectives,
                                                        self.simplified_routes,
                                                        self.StudyAreaXRange,
                                                        self.StudyAreaYRange,
                                                        pop_size)

            
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        
        arcpy.env.workspace = self.arcpyworkspace
#        srs=arcpy.Describe("StudyArea").spatialReference
        arcpy.env.outputCoordinateSystem=self.srs
        RoadCostperMeter = self.RoadCostperMeter
        bufferwidth = self.Roadwidth/2

        def ListtoArray (solution: FloatSolution):
            ABIndi=[]
            for i in range(int(len(solution.variables)/self.n_D)):
                ABIndi.append([solution.variables[self.n_D*i+j] for j in range(self.n_D)])
            return ABIndi

        def ArraytoList (ABIndi):
            LBIndi=[]
            for i in range(len(ABIndi)):
                for j in range(self.n_D):
                    LBIndi.append(ABIndi[i][j])
            return LBIndi

        def Remove_Repetitive_IPs(ABIndi):
            Cleaned_List = []
            for item in ABIndi:
                if not Cleaned_List or item != Cleaned_List[-1]:
                    Cleaned_List.append(item)

            return Cleaned_List

        def PIX (ABIndi,i): #Return X Coordinate of ith PI in the Individual (road path) (meters)
            if i == -1:
                return round(float(self.startpoint[0]),2)
            elif i == len(ABIndi):
                return round(float(self.endpoint[0]),2)
            else:
                return round(float(ABIndi[i][0]),2)
            
        def PIY (ABIndi,i): #Return Y Coordinate of ith PI in the Individual (road path) (meters)
            if i == -1:
                return round(float(self.startpoint[1]),2)
            elif i == len(ABIndi):
                return round(float(self.endpoint[1]),2)
            else:
                return round(float(ABIndi[i][1]),2)

        def VectorX (ABIndi,i): #Return first component of vector which starts in (i-1)th PI and ends in ith PI (meters)
            return (PIX(ABIndi,i)-PIX(ABIndi,i-1))
        def VectorY (ABIndi,i): #Return second component of vector which starts in (i-1)th PI and ends in ith PI (meters)
            return PIY(ABIndi,i)-PIY(ABIndi,i-1)
    #    def VectorZ (ABIndi,i): #Return third component of vector which starts in (i-1)th PI and ends in ith PI (meters)
    #        return PIZ(ABIndi,i)-PIZ(ABIndi,i-1)
        def VectorExteriorProduct (ABIndi,i): #Return Z coordinte of Exterior Product of two vectors that intersect eachother on ith PI
            return round(VectorX(ABIndi,i)*VectorY(ABIndi,i+1)-VectorY(ABIndi,i)*VectorX(ABIndi,i+1),5)
        def DeflectionAngle (ABIndi,i,numpy=numpy): #Return Deflection Angle on ith PI
            return numpy.sign(VectorExteriorProduct(ABIndi,i))*self.lengthofspiral/(6*Radius(ABIndi,i))
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE All Deflection Angles ARE ZERO.####
        def CosinSlope(ABIndi,i): #Return Cosine of angle which is between the vector, starts in (i-1)th PI and ends in ith PI, and X axis
            if VectorX(ABIndi,i) == 0.0:
                return 0.0
            return VectorX(ABIndi,i)/(VectorX(ABIndi,i)**2+VectorY(ABIndi,i)**2)**0.5
        def Slope(ABIndi,i,numpy=numpy):
            return numpy.sign(VectorY(ABIndi,i))*math.acos(CosinSlope(ABIndi,i))
        def DistanceofPI (ABIndi,i): #Distance between (i-1)th PI and ith PI (meters)
            return (VectorX(ABIndi,i)**2+VectorY(ABIndi,i)**2)**0.5
        def LengthofThrow (ABIndi,i): #Return Length of Throw in ith PI (meters)
            if self.lengthofspiral ==0:
                return 0.0
            return self.lengthofspiral**2/(24*Radius(ABIndi,i))
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE All Length of Throws ARE ZERO.####
        def CosinAngleofIntersection(ABIndi,i):
            if VectorExteriorProduct(ABIndi,i) == 0:
                if numpy.sign(VectorX(ABIndi,i))== numpy.sign(VectorX(ABIndi,i+1)):
                    return 1.0
                return 0.0
            else:
                return (VectorX(ABIndi,i)*VectorX(ABIndi,i+1)+VectorY(ABIndi,i)*VectorY(ABIndi,i+1))/(DistanceofPI(ABIndi,i)*DistanceofPI(ABIndi,i+1))
        def AngleofIntersection(ABIndi,i):
            if i== len(ABIndi):
                return math.pi
            cosa = CosinAngleofIntersection(ABIndi,i)
            if cosa>1:
                print('cosa>1:',cosa)
                print(ABIndi)
                return math.acos(1)
            elif cosa<-1:
                print('cosa<-1:',cosa)
                print(ABIndi)
                return math.acos(-1)
            else:
                return math.acos(CosinAngleofIntersection(ABIndi,i))
        def SpiralTangentDistance (ABIndi,i,R): #Return distance between TS and ith PI (meters)
            if i==-1 or i==len(ABIndi):
                return 0
            else:
                return self.lengthofspiral/2 + (LengthofThrow(ABIndi,i)+R)*math.tan(AngleofIntersection(ABIndi,i)/2)
        def Radii (ABIndi): #Returns the list of Radii
#            print("RADII")
            radii = dict(zip(range(-1,len(ABIndi)+1),[0]+[self.Radius_min]*len(ABIndi)+[0]))
            LoTs = dict(zip(range(-1,len(ABIndi)+1),[0]+[SpiralTangentDistance (ABIndi,i,radii[i]) for i in range(len(ABIndi))]+[0]))
            Defs = dict(zip(range(-1,len(ABIndi)),[LoTs[i]+LoTs[i+1]-DistanceofPI (ABIndi,i+1) for i in range(-1,len(ABIndi))]))
            fixed= dict(zip(range(-1,len(ABIndi)+1),[1]+[0]*len(ABIndi)+[1]))

            def Identify(ABIndi, radi):
#                print("identify")
                I_c = 0
                R_c = self.Radius_min
                R_t = self.Radius_min
                IP=-1

                while IP < len(ABIndi):
                    if Defs[IP]>0:
                        if fixed[IP] == fixed[IP+1] == 0:
                            R_t = DistanceofPI (ABIndi,IP+1)/(math.tan(AngleofIntersection(ABIndi,IP)/2)+math.tan(AngleofIntersection(ABIndi,IP+1)/2))
                        if fixed[IP] == 0 and fixed[IP+1] == 1:
                            R_t = (DistanceofPI (ABIndi,IP+1)-radii[IP+1]*math.tan(AngleofIntersection(ABIndi,IP+1)/2))/math.tan(AngleofIntersection(ABIndi,IP)/2)
                        if fixed[IP] == 1 and fixed[IP+1] == 0:
                            R_t = (DistanceofPI (ABIndi,IP+1)-radii[IP]*math.tan(AngleofIntersection(ABIndi,IP)/2))/math.tan(AngleofIntersection(ABIndi,IP+1)/2)
                        if R_t < R_c:
                            R_c =R_t
                            I_c = IP
                    IP = IP+1
                return I_c,R_c
            while True:
#                print('True')
    #        for IP in range(len(ABIndi)):
                I_c,R_c = Identify(ABIndi, radii)
                if fixed[I_c]==fixed[I_c+1]==0:
                    radii[I_c]=R_c
                    radii[I_c+1]=R_c
                    LoTs = dict(zip(range(-1,len(ABIndi)+1),[0]+[SpiralTangentDistance (ABIndi,i,radii[i]) for i in range(len(ABIndi))]+[0]))
                    Defs = dict(zip(range(-1,len(ABIndi)),[LoTs[i]+LoTs[i+1]-DistanceofPI (ABIndi,i+1) for i in range(-1,len(ABIndi))]))
                    fixed[I_c]=1
                    fixed[I_c+1]=1

                elif fixed[I_c]==0 and fixed[I_c+1]==1:
                    radii[I_c]=R_c
                    LoTs = dict(zip(range(-1,len(ABIndi)+1),[0]+[SpiralTangentDistance (ABIndi,i,radii[i]) for i in range(len(ABIndi))]+[0]))
                    Defs = dict(zip(range(-1,len(ABIndi)),[LoTs[i]+LoTs[i+1]-DistanceofPI (ABIndi,i+1) for i in range(-1,len(ABIndi))]))
                    fixed[I_c]=1
                    
                elif fixed[I_c]==1 and fixed[I_c+1]==0:
                    radii[I_c+1]=R_c
                    LoTs = dict(zip(range(-1,len(ABIndi)+1),[0]+[SpiralTangentDistance (ABIndi,i,radii[i]) for i in range(len(ABIndi))]+[0]))
                    Defs = dict(zip(range(-1,len(ABIndi)),[LoTs[i]+LoTs[i+1]-DistanceofPI (ABIndi,i+1) for i in range(-1,len(ABIndi))]))
                    fixed[I_c+1]=1
                if R_c == self.Radius_min:
#                    print(radii)
                    return radii

        ABSolu = ListtoArray (solution)
        ABSolu = Remove_Repetitive_IPs(ABSolu)
#        ABSolu = self.Handling (ABSolu)
#        LBSolu = ArraytoList (ABSolu)
#        solution.variables = self.LBSolu
        Radiilist = Radii (ABSolu)
        
        def Radius (ABIndi,i):#Return Radius of ith PI in the Individual (road path) (meters)
#            print('Radius',i,Radiilist[i])
            return Radiilist[i]
        def OffsetDistance (ABIndi,i): #Return offset distance in ith PI (meters)
            return self.lengthofspiral**2/(6*Radius(ABIndi,i))
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE All Offset Distances ARE ZERO.####
        def DistanceAlongTangent (ABIndi,i): #Return Distance Along Tangent in ith PI (meters)
            return self.lengthofspiral - (self.lengthofspiral**3)/(40*Radius(ABIndi,i)**2)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE All Distance Along Tangent ARE ZERO.####
        def DistancefromTStoSC (ABIndi,i): #Return Distance from TS to SC in ith PI (meters)
            return (DistanceAlongTangent(ABIndi,i)**2+OffsetDistance(ABIndi,i)**2)**0.5
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE All Distance from TS to SC ARE ZERO.####
        def SpiralAnglefromTangenttoSC (ABIndi,i): #Return Spiral Angle from Tangent to SC in ith PI (meters)
            return self.lengthofspiral/(2*Radius(ABIndi,i))
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE All Spiral Angles from Tangent to SC ARE ZERO.####
        def LengthofArc (ABIndi,i): #Return the length of arc in ith PI (meters)
            return (AngleofIntersection(ABIndi,i)-2*SpiralAnglefromTangenttoSC(ABIndi,i))*Radius(ABIndi,i)
        def TSX (ABIndi,i): #Return the X coordinte of TS in ith PI (meters)
           
            if i==len(ABIndi):
                return self.endpoint[0]
            else:
                R = Radius (ABIndi,i)
                return VectorX(ABIndi,i)*(DistanceofPI(ABIndi,i)-SpiralTangentDistance(ABIndi,i,R))/DistanceofPI(ABIndi,i)+PIX(ABIndi,i-1)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE TS and SC are coincident.####
        def TSY (ABIndi,i): #Return the Y coordinte of TS in ith PI (meters)
            if i==len(ABIndi):
                return self.endpoint[1]
            else:
                R = Radius (ABIndi,i)
                return VectorY(ABIndi,i)*(DistanceofPI(ABIndi,i)-SpiralTangentDistance(ABIndi,i,R))/DistanceofPI(ABIndi,i)+PIY(ABIndi,i-1)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE TS and SC are coincident.####
        def SCX (ABIndi,i): #Return the X coordinte of SC in ith PI (meters)
            return TSX(ABIndi,i)+math.cos(DeflectionAngle(ABIndi,i)+Slope(ABIndi,i))*DistancefromTStoSC(ABIndi,i)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE TS and SC are coincident.####
        def SCY (ABIndi,i): #Return the Y coordinte of SC in ith PI (meters)
            return TSY(ABIndi,i)+math.sin(DeflectionAngle(ABIndi,i)+Slope(ABIndi,i))*DistancefromTStoSC(ABIndi,i)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE TS and SC are coincident.####
        def STX (ABIndi,i): #Return the X coordinte of ST in ith PI (meters)
            if i==-1:
                return self.startpoint[0]
            else:
                R = Radius (ABIndi,i)
                return VectorX(ABIndi,i+1)*SpiralTangentDistance(ABIndi,i,R)/DistanceofPI(ABIndi,i+1)+PIX(ABIndi,i)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE ST and CS are coincident.####
        def STY (ABIndi,i): #Return the Y coordinte of ST in ith PI (meters)
            if i==-1:
                return self.startpoint[1]
            else:
                R = Radius (ABIndi,i)
                return VectorY(ABIndi,i+1)*SpiralTangentDistance(ABIndi,i,R)/DistanceofPI(ABIndi,i+1)+PIY(ABIndi,i)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE ST and CS are coincident.####
        def CSX (ABIndi,i): #Return the X coordinte of CS in ith PI (meters)
            return STX(ABIndi,i)+math.cos(math.pi+Slope(ABIndi,i+1)-DeflectionAngle(ABIndi,i))*DistancefromTStoSC(ABIndi,i)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE ST and CS are coincident.####
        def CSY (ABIndi,i): #Return the Y coordinte of CS in ith PI (meters)
            return STY(ABIndi,i)+math.sin(math.pi+Slope(ABIndi,i+1)-DeflectionAngle(ABIndi,i))*DistancefromTStoSC(ABIndi,i)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE ST and CS are coincident.####
        def DistancefromSCtoCS (ABIndi,i): #Return distance between SC and CS in ith PI (meters)
            return math.sqrt((SCX(ABIndi,i)-CSX(ABIndi,i))**2+(SCY(ABIndi,i)-CSY(ABIndi,i))**2)
        def COAX (ABIndi,i,numpy=numpy): #Return the X coordinte of Centre Of Arc in ith PI (meters)
            if VectorExteriorProduct(ABIndi,i) == 0:
                return 1.0
            return (SCX(ABIndi,i)+CSX(ABIndi,i))/2+numpy.sign(VectorExteriorProduct(ABIndi,i))*math.sqrt(abs(Radius(ABIndi,i)**2-(DistancefromSCtoCS(ABIndi,i)/2)**2))*(SCY(ABIndi,i)-CSY(ABIndi,i))/DistancefromSCtoCS(ABIndi,i)
        def COAY (ABIndi,i,numpy=numpy): #Return the Y coordinte of Centre Of Arc (COA) in ith PI (meters)
            if VectorExteriorProduct(ABIndi,i) == 0:
                return 1.0
            return (SCY(ABIndi,i)+CSY(ABIndi,i))/2+numpy.sign(VectorExteriorProduct(ABIndi,i))*math.sqrt(abs(Radius(ABIndi,i)**2-(DistancefromSCtoCS(ABIndi,i)/2)**2))*(CSX(ABIndi,i)-SCX(ABIndi,i))/DistancefromSCtoCS(ABIndi,i)
        def SCAngle (ABIndi,i): #Return the angle between COA-SC vector and x axis (degrees) in ith PI (degrees)
            return (180*math.atan2(SCY(ABIndi,i)-COAY(ABIndi,i),SCX(ABIndi,i)-COAX(ABIndi,i)))/math.pi
        def CSAngle (ABIndi,i): #Return the angle between COA-CS vector and x axis (degrees) in ith PI (degrees)
            return (180*math.atan2(CSY(ABIndi,i)-COAY(ABIndi,i),CSX(ABIndi,i)-COAX(ABIndi,i)))/math.pi
        def StartAngleofArc (ABIndi,i): #Return the angle that the arc starts counterclockwisely in ith PI (degrees)
            if VectorExteriorProduct(ABIndi,i)>0:
                return SCAngle(ABIndi,i)
            else:
                return CSAngle(ABIndi,i)
        def EndAngleofArc (ABIndi,i): #Return the angle that the arc ends counterclockwisely in ith PI (degrees)
            if VectorExteriorProduct(ABIndi,i)>0:
                return CSAngle(ABIndi,i)
            else:
                return SCAngle(ABIndi,i)
        def TSKM (ABIndi,i): #Return the distance along road from start point to TS in ith PI (meters)
#            print("TSKM", i)
            if i==len(ABIndi):
                return STKM(ABIndi,len(ABIndi)-1)+((self.endpoint[0]-STX(ABIndi,len(ABIndi)-1))**2+(self.endpoint[1]-STY(ABIndi,len(ABIndi)-1))**2)**0.5
            else:
                return math.sqrt((TSX(ABIndi,i)-STX(ABIndi,i-1))**2+(TSY(ABIndi,i)-STY(ABIndi,i-1))**2)+STKM(ABIndi,i-1)
        def SCKM (ABIndi,i): #Return the distance along road from start point to SC in ith PI (meters)
#            print("SCKM", i)
            return TSKM(ABIndi,i)+self.lengthofspiral

        def CSKM (ABIndi,i): #Return the distance along road from start point to CS in ith PI (meters)
#            print("CSKM", i)
            return SCKM(ABIndi,i)+LengthofArc(ABIndi,i)
            
        def STKM (ABIndi,i): #Return the distance along road from start point to ST in ith PI (meters)
#            print("STKM", i)
            if i==-1:
                return 0
            else:
                return CSKM(ABIndi,i)+self.lengthofspiral
        def PIKM (ABIndi,i): #Return the distance along road from start point to ith PI (meters)
#            print("PIKM", i)
            R = Radius (ABIndi,i)
            if i==len(ABIndi):
                return STKM(ABIndi,len(ABIndi)-1)+((self.endpoint[0]-STX(ABIndi,len(ABIndi)-1))**2+(self.endpoint[1]-STY(ABIndi,len(ABIndi)-1))**2)**0.5
            else:
                return TSKM(ABIndi,i)+SpiralTangentDistance(ABIndi,i,R)
        def EndPointKM (ABIndi): #Return the length of road called Individual
            return STKM(ABIndi,len(ABIndi)-1)+((self.endpoint[0]-STX(ABIndi,len(ABIndi)-1))**2+(self.endpoint[1]-STY(ABIndi,len(ABIndi)-1))**2)**0.5
        def DeflectionAnglefromTStoAnyPointofFirstSpiral(ABIndi,i,KM): #Return Deflection Angle from TS to any point which its distance from start point is KM on first spiral on ith PI
            return (KM-TSKM(ABIndi,i))**2/(6*Radius(ABIndi,i)*self.lengthofspiral)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE all deflection angles from TS to any point on first spiral are ZERO.####
        def DistanceAlongTangenttoAnyPointofFirstSpiral(ABIndi,i,KM): #Return distance along first spiral from TS to any point which its distance from start point is KM on first spiral on ith PI
            return (KM-TSKM(ABIndi,i))-((KM-TSKM(ABIndi,i))**5/(40*Radius(ABIndi,i)**2*self.lengthofspiral**2))
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE all distances along first spiral from TS to any point on first spiral are ZERO.####
        def OffsetDistancefromTangenttoanyPointofFirstSpiral(ABIndi,i,KM): #Return Offset Distance from Tangent to any Point of First Spiral in ith PI (meters)
            return (KM-TSKM(ABIndi,i))**3/(6*Radius(ABIndi,i)*self.lengthofspiral)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE THIS FUNCTION ALWAYS RETURNS ZERO.####
        def TStoAnyPointofSpiralKM (ABIndi,i,KM): #Return direct distance between TS and Any Point of Spiral which its distance from start point along road is KM in ith PI (meters)
            return ((DistanceAlongTangenttoAnyPointofFirstSpiral(ABIndi,i,KM)**2+OffsetDistancefromTangenttoanyPointofFirstSpiral(ABIndi,i,KM)**2)**0.5)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE THIS FUNCTION ALWAYS RETURNS ZERO.####
        def DeflectionAnglefromTStoAnyPointofSecondSpiral(ABIndi,i,KM): #Return Deflection Angle from ST to any point which its distance from start point is KM on second spiral on ith PI
            return (KM-STKM(ABIndi,i))**2/(6*Radius(ABIndi,i)*self.lengthofspiral)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE THIS FUNCTION ALWAYS RETURNS ZERO.####
        def DistanceAlongTangenttoAnyPointofSecondSpiral(ABIndi,i,KM): #Return distance along second spiral from ST to any point which its distance from start point is KM on second spiral on ith PI
            return (STKM(ABIndi,i)-KM)-(STKM(ABIndi,i)-KM)**5/(40*Radius(ABIndi,i)**2*self.lengthofspiral**2)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE THIS FUNCTION ALWAYS RETURNS ZERO.####
        def OffsetDistancefromTangenttoanyPointofSecondSpiral(ABIndi,i,KM): #Return Offset Distance from Tangent to any Point of Second Spiral in ith PI (meters)
            return (STKM(ABIndi,i)-KM)**3/(6*Radius(ABIndi,i)*self.lengthofspiral)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE THIS FUNCTION ALWAYS RETURNS ZERO.####
        def STtoAnyPointofSpiralKM (ABIndi,i,KM): #Return direct distance between ST and Any Point of Spiral which its distance from start point along road is KM in ith PI (meters)
            return ((DistanceAlongTangenttoAnyPointofSecondSpiral(ABIndi,i,KM)**2+OffsetDistancefromTangenttoanyPointofSecondSpiral(ABIndi,i,KM)**2)**0.5)
        ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE THIS FUNCTION ALWAYS RETURNS ZERO.####

            #Functions below are defined to check if road called ABIndi has self intersection#

        def ccw (A,B,C):
            return (C[1]-A[1])*(B[0]-A[0])>(B[1]-A[1])*(C[0]-A[0])
        def intersect (seg1, seg2): #Return True if segment A-B intersects segment C-D
            (A,B),(C,D) = seg1, seg2
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        def linesegments (ABIndi): #Return a list of all segments which start from (i-1)th PI and ends on ith PI
            return [[[PIX(ABIndi,i-1),PIY(ABIndi,i-1)],[PIX(ABIndi,i),PIY(ABIndi,i)]] for i in range(len(ABIndi)+1)]
        def SelfIntersection (ABIndi):
            line_segments = linesegments(ABIndi)
            intersections=[]
            for i in range(len(line_segments)):
                for j in range(i-1):
                    if intersect(line_segments[i],line_segments[j]):
                        intersections.append(-1) #SelfIntersected
                        intersections.append((i,j))
                    else:
                        intersections.append(1)
            if all(i==1 for i in intersections):
                return 1
            else:
                return -1

            #Function below is defined to convert list based road chromosome to array based chromosome#
        
        def Handling (ABIndi): #THIS FUNCTION HANDLES INDIVIDUALS INTERSECT THEMSELVES (USED IN FIRST POPULATION)

#            print("handling")
            ind = ABIndi
            r=0
            while SelfIntersection(ind)==-1 and r< 2000:
                
                r=r+1
                line_segments = linesegments(ind)
                for k in range(len(ind)):
                    for m in range(k+2,len(ind)+1):
                        if intersect(line_segments[k],line_segments[m]):
                            ind[k] , ind[m-1] = ind[m-1], ind[k]
                            line_segments = linesegments(ind)

            return ind

    ##############
        #################

        TSXlist=[TSX(ABSolu,i) for i in range(len(ABSolu)+1)]                                #List containing all X coordinates of TSs
        TSYlist=[TSY(ABSolu,i) for i in range(len(ABSolu)+1)]                                #List containing all Y coordinates of TSs
        TSKMlist=[TSKM(ABSolu,i) for i in range(len(ABSolu)+1)]                              #List containing all distances between starting point and TS along road
        SCXlist=[SCX(ABSolu,i) for i in range(len(ABSolu))]
        SCYlist=[SCY(ABSolu,i) for i in range(len(ABSolu))]
        SCKMlist=[SCKM(ABSolu,i) for i in range(len(ABSolu))]
        CSXlist=[CSX(ABSolu,i) for i in range(len(ABSolu))]
        CSYlist=[CSY(ABSolu,i) for i in range(len(ABSolu))]
        CSKMlist=[CSKM(ABSolu,i) for i in range(len(ABSolu))]
        STXlist=[STX(ABSolu,i) for i in list(range(len(ABSolu)))+[-1]]
        STYlist=[STY(ABSolu,i) for i in list(range(len(ABSolu)))+[-1]]
        STKMlist=[STKM(ABSolu,i) for i in list(range(len(ABSolu)))+[-1]]
        COAXlist=[COAX(ABSolu,i) for i in range(len(ABSolu))]
        COAYlist=[COAY(ABSolu,i) for i in range(len(ABSolu))]
            
        PointstoDrawX=[self.startpoint[0]]   #List containing all X coordinates of points needed to draw the centerline of road
        PointstoDrawY=[self.startpoint[1]]   #List containing all Y coordinates of points needed to draw the centerline of road
        
        for j in range(0,len(ABSolu)):  #Generate X and Y coordinates of centreline 
            
            StationKM=numpy.arange(TSKMlist[j],STKMlist[j],self.roadaccuracy)
            if not StationKM.size == 0:
                StationKM.put(-1,STKMlist[j])
            for i in StationKM:
                if i<=TSKMlist[j]:
                    X1=STXlist[j-1]+math.cos(Slope(ABSolu,j))*(i-STKMlist[j-1])
                    PointstoDrawX.append(X1)
                    
                    Y1=STYlist[j-1]+math.sin(Slope(ABSolu,j))*(i-STKMlist[j-1])
                    PointstoDrawY.append(Y1)
    ####IN THIS RESEARCH WE ASSUMED ALL ROAD BENDS ARE SIMPLE ARCS. THERFORE CONDITION BELOW NEVER WILL BE TRUE.####
                elif TSKMlist[j]<i<SCKMlist[j]:
                    #print ABSolu
                    PointstoDrawX.append(TSXlist[j]+math.cos(Slope(ABSolu,j)+numpy.sign(VectorExteriorProduct(ABSolu,j))*DeflectionAnglefromTStoAnyPointofFirstSpiral(ABSolu,j,i))*(TStoAnyPointofSpiralKM(ABSolu,j,i)))
                    PointstoDrawY.append(TSYlist[j]+math.sin(Slope(ABSolu,j)+numpy.sign(VectorExteriorProduct(ABSolu,j))*DeflectionAnglefromTStoAnyPointofFirstSpiral(ABSolu,j,i))*(TStoAnyPointofSpiralKM(ABSolu,j,i)))

                elif SCKMlist[j]<=i<=CSKMlist[j]:
                    PointstoDrawX.append(COAXlist[j]+(SCXlist[j]-COAXlist[j])*math.cos(numpy.sign(VectorExteriorProduct(ABSolu,j))*(i-SCKMlist[j])/Radius(ABSolu,j))-(SCYlist[j]-COAYlist[j])*math.sin(numpy.sign(VectorExteriorProduct(ABSolu,j))*(i-SCKMlist[j])/Radius(ABSolu,j)))
                    PointstoDrawY.append(COAYlist[j]+(SCXlist[j]-COAXlist[j])*math.sin(numpy.sign(VectorExteriorProduct(ABSolu,j))*(i-SCKMlist[j])/Radius(ABSolu,j))+(SCYlist[j]-COAYlist[j])*math.cos(numpy.sign(VectorExteriorProduct(ABSolu,j))*(i-SCKMlist[j])/Radius(ABSolu,j)))

                elif CSKMlist[j]<i<STKMlist[j]:
                    #print ABSolu
                    PointstoDrawX.append(STXlist[j]+math.cos(math.pi+Slope(ABSolu,j+1)-numpy.sign(VectorExteriorProduct(ABSolu,j))*DeflectionAnglefromTStoAnyPointofSecondSpiral(ABSolu,j,i))*STtoAnyPointofSpiralKM(ABSolu,j,i))
                    PointstoDrawY.append(STYlist[j]+math.sin(math.pi+Slope(ABSolu,j+1)-numpy.sign(VectorExteriorProduct(ABSolu,j))*DeflectionAnglefromTStoAnyPointofSecondSpiral(ABSolu,j,i))*STtoAnyPointofSpiralKM(ABSolu,j,i))

                else:
                    X2=STXlist[j]+math.cos(Slope(ABSolu,j+1))*(i-STKMlist[j])
                    PointstoDrawX.append(X2)
                    
                    Y2=STYlist[j]+math.sin(Slope(ABSolu,j+1))*(i-STKMlist[j])
                    PointstoDrawY.append(Y2)

            
        PointstoDrawX.append(self.endpoint[0])
        PointstoDrawY.append(self.endpoint[1])
#        print ('PointstoDrawY',PointstoDrawY)
        PointstoDrawXYZ=zip(PointstoDrawX,PointstoDrawY)
        IndividualName='I1'+str(os.getpid())+str(os.getpid())  #ASSIGN A UNIQUE NAME TO INDIVIDUAL SINCE WE USE MULTIPROCESSING LIBRARY)
        processnumber=1
        
        while arcpy.Exists('in_memory\\'+IndividualName):
            processornumber=processornumber+1
            IndividualName='I'+str(processnumber)+str(os.getpid())
            
        arcpy.env.workspace = self.arcpyworkspace
        srs=arcpy.Describe("StudyArea").spatialReference
        arcpy.env.outputCoordinateSystem=srs
        arcpy.env.outputZFlag = "DISABLED"

        if not arcpy.Exists(r'in_memory\ALLinmemory'):
            arcpy.CopyFeatures_management(r'ALL',r'in_memory\ALLinmemory')

        if not arcpy.Exists(r'in_memory\Parcelsinmemory'):
            arcpy.CopyFeatures_management(r'Parcels',r'in_memory\Parcelsinmemory')

        Centerline = arcpy.CreateFeatureclass_management(out_path='in_memory',out_name=IndividualName,geometry_type="POLYLINE", has_m= "DISABLED",has_z= "DISABLED" ,spatial_reference=self.srs)
        cursor = arcpy.da.InsertCursor(Centerline, ["SHAPE@"])
        lArray=[]
        for point in PointstoDrawXYZ:
            lArray.append(arcpy.Point(X=point[0],Y=point[1]))
        array = arcpy.Array(lArray)
        polyline=arcpy.Polyline(array)
        cursor.insertRow([polyline])
        del cursor
        todelete = [r'in_memory\\'+IndividualName]
        RouteLength = arcpy.da.FeatureClassToNumPyArray('in_memory\\'+IndividualName,'SHAPE@LENGTH')[0][0]

                                            #CREATE AN OBJECT OF INFRASTRUCTURE (RPGN) IN GIS ENVIRONMENT
        arcpy.Buffer_analysis(Centerline,'in_memory\Buffered'+IndividualName,'%s Meters' %str(bufferwidth),'FULL','FLAT','NONE')
        todelete.append(r'in_memory\Buffered'+IndividualName)
        RoadArea = arcpy.da.FeatureClassToNumPyArray('in_memory\Buffered'+IndividualName,'SHAPE@AREA')['SHAPE@AREA'].sum()

        arcpy.AddField_management(r'in_memory\Buffered'+IndividualName,"TEST","SHORT")
        arcpy.CalculateField_management (r'in_memory\Buffered'+IndividualName,"TEST","1", "PYTHON_9.3")
        arcpy.TabulateIntersection_analysis(r'in_memory\Buffered'+IndividualName,"TEST",r'in_memory\ALLinmemory',r'in_memory\outtable'+IndividualName,'ZONE')
        todelete.append(r'in_memory\outtable'+IndividualName)
        
        Intersections=arcpy.da.TableToNumPyArray(r'in_memory\outtable'+IndividualName,('ZONE',"Area"))

        MV = dict(Intersections)
        for inter in self.All_Intersections:
            if not inter in MV:
                MV[inter]=0
        

        #Define Importance of Violation
        IV={}
        
        if MV['Wetland']>0:
            IV['VZ_NW12']=16

        else:
##            arcpy.env.workspace = self.arcpyworkspace
##            srs=arcpy.Describe("StudyArea").spatialReference
##            arcpy.env.outputCoordinateSystem=srs
##            arcpy.env.outputZFlag = "DISABLED"
#            print(arcpy.Exists(r'Wetland'))
            if not arcpy.Exists('Wetland_Layer'):
                arcpy.MakeFeatureLayer_management (r'Wetland','Wetland_Layer')
            arcpy.SelectLayerByLocation_management ('Wetland_Layer',"WITHIN_A_DISTANCE",r'in_memory\Buffered'+IndividualName,'15 Meters','NEW_SELECTION')
            matchcount = int(arcpy.GetCount_management('Wetland_Layer')[0])
            if matchcount>0:
                IV['VZ_NW12']=16
            else:
                arcpy.SelectLayerByLocation_management ('Wetland_Layer',"WITHIN_A_DISTANCE",r'in_memory\Buffered'+IndividualName,'30 Meters','NEW_SELECTION')
                matchcount = int(arcpy.GetCount_management('Wetland_Layer')[0])
                if matchcount>0:
                    IV['VZ_NW12']=14
                else:
                    arcpy.SelectLayerByLocation_management ('Wetland_Layer',"WITHIN_A_DISTANCE",r'in_memory\Buffered'+IndividualName,'45 Meters','NEW_SELECTION')
                    matchcount = int(arcpy.GetCount_management('Wetland_Layer')[0])
                    if matchcount>0:
                        IV['VZ_NW12']=11
                    else:
                        arcpy.SelectLayerByLocation_management ('Wetland_Layer',"WITHIN_A_DISTANCE",r'in_memory\Buffered'+IndividualName,'60 Meters','NEW_SELECTION')
                        matchcount = int(arcpy.GetCount_management('Wetland_Layer')[0])
                        if matchcount>0:
                            IV['VZ_NW12']=6
                        else:
                            IV['VZ_NW12']=0

        #NW1.3
        if 0.1<= MV['VZ_NW13']/RoadArea:
            IV['VZ_NW13']=12
        elif 0.05<= MV['VZ_NW13']/RoadArea<0.1:
            IV['VZ_NW13']=10
        elif 0.0<MV['VZ_NW13']/RoadArea<0.05:
            IV['VZ_NW13']=4
        else:
            IV['VZ_NW13']=0


        S_V = sum([IV[i]*MV[i] for i in self.CR])
        
        C_L = RouteLength*RoadCostperMeter

        arcpy.TabulateIntersection_analysis (r'in_memory\Buffered'+IndividualName, 'TEST' , r'in_memory\Parcelsinmemory', 'in_memory\parcels_%s' %IndividualName, 'LandValuePerUnit')
        todelete.append('in_memory\parcels_%s' %IndividualName)

        Parcel_fields = arcpy.da.TableToNumPyArray('in_memory\parcels_%s' %IndividualName, ('LandValuePerUnit', 'Area'))
        C_R=sum([i[0]*i[1] for i in Parcel_fields])

        C_N=sum(self.UnitLocationDependentCost[i]*MV[i] for i in self.UnitLocationDependentCost)
        #Calculate Penalty Cost
        C_P = 0
        if not len(ABSolu)==0:
            C_P = sum([self.Alpha0+self.Alpha1*(self.Radius_min-Radius(ABSolu,i))**self.Alpha2 for i in range(len(ABSolu)) if self.Radius_min-Radius(ABSolu,i)>0]) #/len(ABIndi)
        C_T = C_L + C_R + C_N + C_P

        for i in todelete:
            arcpy.Delete_management(i)
        
        solution.objectives[0] = S_V
        solution.objectives[1] = C_T
#        print(solution.objectives)
        solution.C_L = C_L
        solution.C_R = C_R
        solution.C_N = C_N
        solution.C_P = C_P
        solution.IV = IV
        solution.MV = MV
        
        return solution
    
    def name(self):
        return "Alignment_Problem"

    def create_solution(self) -> FloatSolution:
        if len(self.initial_solutions) == 0:
            self.initial_solutions = generate_initial_sols (arcpyworkspace, self.number_of_variables(), self.number_of_objectives(), self.simplified_routes, self.StudyAreaXRange, self.StudyAreaYRange, self.pop_size)
#        print(len(self.initial_solutions))
        sol = self.initial_solutions.pop(0)
        return sol


##if __name__ == "__main__":
##    
##    problem = Alignment_Problem(16,2,False)
##    max_evaluations = 400
##    algorithm = NSGAII(
##        problem=problem,
##        population_size=100,
##        offspring_population_size=100,
##        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
##        crossover=SBXCrossover(probability=1.0, distribution_index=20),
##        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
##        population_evaluator=MultiprocessEvaluator(),
##    )
##    
##    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
##    algorithm.observable.register(observer=VisualizerObserver())
##
##    algorithm.run()
##    front = get_non_dominated_solutions(algorithm.get_result())
##
##    # Save results to file
##    print_function_values_to_file(front, "FUN." + algorithm.label)
##    print_variables_to_file(front, "VAR." + algorithm.label)
##
##    print(f"Algorithm: {algorithm.get_name()}")
##    print(f"Problem: {problem.name()}")
##    print(f"Computing time: {algorithm.total_computing_time}")
##
##    plot_front = Plot(
##        title="Pareto front approximation. Problem: " + problem.name(),
##        axis_labels=problem.obj_labels,
##    )
##    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())
##
##    # Plot interactive front
##    plot_front = InteractivePlot(
##        title="Pareto front approximation. Problem: " + problem.name(),
##        axis_labels=problem.obj_labels,
##    )
##    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())
