import numpy as np
from lxml import etree
import os

'''
Does not yet include signaling parameters
'''

class VXA:
    
    def __init__(self, HeapSize=0.1, PrintfFIFOSize=50, EnableCilia=0, EnableExpansion=0, DtFrac=0.95, BondDampingZ=1, ColDampingZ=0.8,
                SlowDampingZ=0.01, EnableCollision=0, SimTime=5, TempPeriod=0, GravEnabled=1, GravAcc=-9.81, FloorEnabled=1, Lattice_Dim=0.01,
                RecordStepSize=100, RecordVoxel=1, RecordLink=0, RecordFixedVoxels=1, VaryTempEnabled=0, TempAmplitude=0, TempBase=0,
                TempEnabled=0):

        root = etree.XML("<VXA></VXA>")
        root.set('Version', '1.1')
        self.tree = etree.ElementTree(root)

        self.HeapSize = HeapSize
        self.PrintfFIFOSize = PrintfFIFOSize
        self.EnableCilia = EnableCilia
        self.EnableExpansion = EnableExpansion
        self.DtFrac = DtFrac
        self.BondDampingZ = BondDampingZ
        self.ColDampingZ = ColDampingZ
        self.SlowDampingZ = SlowDampingZ
        self.EnableCollision = EnableCollision
        self.SimTime = SimTime
        self.TempPeriod = TempPeriod
        self.GravEnabled = GravEnabled
        self.GravAcc = GravAcc
        self.FloorEnabled = FloorEnabled
        self.Lattice_Dim = Lattice_Dim
        self.RecordStepSize = RecordStepSize
        self.RecordVoxel = RecordVoxel
        self.RecordLink = RecordLink
        self.RecordFixedVoxels = RecordFixedVoxels
        self.VaryTempEnabled = VaryTempEnabled
        self.TempAmplitude = TempAmplitude
        self.TempBase = TempBase
        self.TempEnabled = TempEnabled
        
        self.NextMaterialID = 1 # Material ID's start at 1, 0 denotes empty space

        self.set_default_tags()

    def set_default_tags(self):
        root = self.tree.getroot()

        # GPU
        gpu = etree.SubElement(root, 'GPU')
        etree.SubElement(gpu, "HeapSize").text = str(self.HeapSize)
        # etree.SubElement(gpu, "PrintfFIFOSize").text = str(self.PrintfFIFOSize)
        
        # Simulator
        simulator = etree.SubElement(root, "Simulator")
        etree.SubElement(simulator, "EnableCilia").text = str(self.EnableCilia)
        etree.SubElement(simulator, "EnableExpansion").text = str(self.EnableExpansion) # 0 only contraction, 1 is contration + expansion

        integration = etree.SubElement(simulator, "Integration")
        etree.SubElement(integration, "DtFrac").text = str(self.DtFrac)

        damping = etree.SubElement(simulator, "Damping")
        etree.SubElement(damping, "BondDampingZ").text = str(self.BondDampingZ)
        etree.SubElement(damping, "ColDampingZ").text = str(self.ColDampingZ)
        etree.SubElement(damping, "SlowDampingZ").text = str(self.SlowDampingZ)

        attachDetach = etree.SubElement(simulator, "AttachDetach")
        etree.SubElement(attachDetach, "EnableCollision").text = str(self.EnableCollision)

        stopCondition = etree.SubElement(simulator, "StopCondition")
        formula = etree.SubElement(stopCondition, "StopConditionFormula")
        sub = etree.SubElement(formula, "mtSUB")
        etree.SubElement(sub, "mtVAR").text = 't'
        etree.SubElement(sub, "mtCONST").text = str(self.SimTime)

        # fitness = etree.SubElement(simulator, "FitnessFunction") # default - maximum x distance
        # add = etree.SubElement(fitness, "mtADD")
        # mul = etree.SubElement(add, 'mtMUL')
        # etree.SubElement(mul, "mtVAR").text = 'x'
        # etree.SubElement(mul, "mtVAR").text = 'x'
        # mul2 = etree.SubElement(add, 'mtMUL')
        # etree.SubElement(mul2, "mtVAR").text = 'y'
        # etree.SubElement(mul2, "mtVAR").text = 'y'


        history = etree.SubElement(simulator, "RecordHistory")
        etree.SubElement(history, "RecordStepSize").text = str(self.RecordStepSize) #Capture image every 100 time steps
        etree.SubElement(history, "RecordVoxel").text = str(self.RecordVoxel) # Add voxels to the visualization
        etree.SubElement(history, "RecordLink").text = str(self.RecordLink) # Add links to the visualization
        etree.SubElement(history, "RecordFixedVoxels").text = str(self.RecordFixedVoxels) 
        
        # Environment

        environment = etree.SubElement(root, "Environment")
        thermal = etree.SubElement(environment, "Thermal")
        etree.SubElement(thermal, "TempEnabled").text = str(self.TempEnabled)
        etree.SubElement(thermal, "VaryTempEnabled").text = str(self.VaryTempEnabled)
        etree.SubElement(thermal, "TempPeriod").text = str(self.TempPeriod)
        etree.SubElement(thermal, "TempAmplitude").text = str(self.TempAmplitude)
        etree.SubElement(thermal, "TempBase").text = str(self.TempBase)

        gravity = etree.SubElement(environment, "Gravity")
        etree.SubElement(gravity, "GravEnabled").text = str(self.GravEnabled)
        etree.SubElement(gravity, "GravAcc").text = str(self.GravAcc)
        etree.SubElement(gravity, "FloorEnabled").text = str(self.FloorEnabled)

        # VXC tags
        vxc = etree.SubElement(root, "VXC")
        vxc.set("Version", "0.94")

        lattice = etree.SubElement(vxc, "Lattice")
        etree.SubElement(lattice, "Lattice_Dim").text = str(self.Lattice_Dim)

        # Materials
        palette = etree.SubElement(vxc, "Palette")

        # Structure
        structure = etree.SubElement(vxc, "Structure")
        structure.set("Compression", "ASCII_READABLE")
        # set some default data
        etree.SubElement(structure, "X_Voxels").text = "1"
        etree.SubElement(structure, "Y_Voxels").text = "1"
        etree.SubElement(structure, "Z_Voxels").text = "2"

        data = etree.SubElement(structure, "Data")
        etree.SubElement(data, "Layer").text = etree.CDATA("0")
        etree.SubElement(data, "Layer").text = etree.CDATA("1")

    def add_material(self, E=100000, RHO=1000, P=0.35, CTE=0, uStatic=1, uDynamic=0.8,
                      isSticky=0, Cilia=0, isBreakable=0, isMeasured=1,
                      RGBA=None, isFixed=0, TempPhase=0, Buoyancy=0, LockZ=0, Fail_Stress=0,
                      Name=None, WaterLevel=None):

        material_ID = self.NextMaterialID
        self.NextMaterialID+=1

        if RGBA is None:
        # assign the material a random color
            RGBA = np.around((np.random.random(), np.random.random(), np.random.random(), 1), 2)
        else:
            if len(RGBA)==3: # if no alpha, add alpha of 255
                RGBA = (RGBA[0],RGBA[1],RGBA[2],1)
            else:
                RGBA = (RGBA[0],RGBA[1],RGBA[2],RGBA[3])

        palette = self.tree.find("*/Palette")
        material = etree.SubElement(palette, "Material")
        
        if Name is not None:
            etree.SubElement(material, "Name").text = Name
        else:
            etree.SubElement(material, "Name").text = str(material_ID)

        display = etree.SubElement(material, "Display")
        etree.SubElement(display, "Red").text = str(RGBA[0])
        etree.SubElement(display, "Green").text = str(RGBA[1])
        etree.SubElement(display, "Blue").text = str(RGBA[2])
        etree.SubElement(display, "Alpha").text = str(RGBA[3])

        mechanical = etree.SubElement(material, "Mechanical")
        etree.SubElement(mechanical, "isMeasured").text = str(isMeasured) # if material should be included in fitness function
        if WaterLevel is not None:
            etree.SubElement(mechanical, "WaterLevel").text = str(WaterLevel) # after this many voxels in height, gravity (increasing with the square of the height above the water level) is applied
        etree.SubElement(mechanical, "Buoyancy").text = str(Buoyancy)
        etree.SubElement(mechanical, "Fixed").text = str(isFixed)
        etree.SubElement(mechanical, "sticky").text = str(isSticky)
        etree.SubElement(mechanical, "Cilia").text = str(Cilia)
        etree.SubElement(mechanical, "LockZ").text = str(LockZ)
        etree.SubElement(mechanical, "MatModel").text = str(isBreakable) # 0 = no failing
        etree.SubElement(mechanical, "Fail_Stress").text = str(Fail_Stress) # 0 if MatModel=0
        etree.SubElement(mechanical, "Elastic_Mod").text = str(E)
        etree.SubElement(mechanical, "Fail_Stress").text = "0" # no fail if matModel is 0
        etree.SubElement(mechanical, "Density").text = str(RHO)
        etree.SubElement(mechanical, "Poissons_Ratio").text = str(P)
        etree.SubElement(mechanical, "CTE").text = str(CTE)
        etree.SubElement(mechanical, "MaterialTempPhase").text = str(TempPhase)
        etree.SubElement(mechanical, "uStatic").text = str(uStatic)
        etree.SubElement(mechanical, "uDynamic").text = str(uDynamic)

        return material_ID

    def write(self, filename='base.vxa'):

        # If no material has been added, add default material
        if self.NextMaterialID==0:
            self.add_material()
        
        with open(filename, 'w+') as f:
            f.write(etree.tostring(self.tree, encoding="unicode", pretty_print=True))