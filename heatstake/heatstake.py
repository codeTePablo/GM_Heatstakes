"""
DETECTOR DE HEAT STAKES - Clustering de Cilindros (Optimizado para piezas ensambladas)
Detecta heat stakes en archivos STEP usando clustering de cilindros.

------------------------------------------------------------
Developer: Pablo
Contacto:  codewithpablo@gmail.com
------------------------------------------------------------
"""

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                               GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                               GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
                               GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
                               GeomAbs_OtherSurface)
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.gp import gp_Pnt
from collections import defaultdict
import numpy as np
import time


class STEPFileAnalyzer:
    """
    Analizador completo de archivos STEP
    Proporciona diagnóstico detallado de la estructura y geometría
    """
    
    def __init__(self, step_file):
        self.step_file = step_file
        self.shape = None
        self.analysis_results = {}
        
    def load_step_file(self):
        """Carga el archivo STEP"""
        print(f"\n{'='*75}")
        print(f"ANALIZADOR DE ARCHIVO STEP")
        print(f"{'='*75}")
        print(f"Archivo: {self.step_file}\n")
        
        print("📂 Cargando archivo STEP...")
        start_time = time.time()
        
        reader = STEPControl_Reader()
        status = reader.ReadFile(self.step_file)
        
        if status != IFSelect_RetDone:
            print(f"❌ Error: No se pudo leer el archivo")
            return False
        
        reader.TransferRoots()
        self.shape = reader.OneShape()
        
        load_time = time.time() - start_time
        print(f"✓ Archivo cargado en {load_time:.2f} segundos\n")
        return True
    
    def analyze_topology(self):
        """Analiza la estructura topológica completa"""
        if self.shape is None:
            print("❌ Error: Primero debe cargar el archivo")
            return None
        
        print("🔍 ANÁLISIS TOPOLÓGICO")
        print("─" * 75)
        
        # Contar elementos topológicos
        explorer_solid = TopExp_Explorer(self.shape, TopAbs_SOLID)
        explorer_shell = TopExp_Explorer(self.shape, TopAbs_SHELL)
        explorer_face = TopExp_Explorer(self.shape, TopAbs_FACE)
        explorer_edge = TopExp_Explorer(self.shape, TopAbs_EDGE)
        explorer_vertex = TopExp_Explorer(self.shape, TopAbs_VERTEX)
        
        num_solids = 0
        while explorer_solid.More():
            num_solids += 1
            explorer_solid.Next()
        
        num_shells = 0
        while explorer_shell.More():
            num_shells += 1
            explorer_shell.Next()
        
        num_faces = 0
        while explorer_face.More():
            num_faces += 1
            explorer_face.Next()
        
        num_edges = 0
        while explorer_edge.More():
            num_edges += 1
            explorer_edge.Next()
        
        num_vertices = 0
        while explorer_vertex.More():
            num_vertices += 1
            explorer_vertex.Next()
        
        topology = {
            'num_solids': num_solids,
            'num_shells': num_shells,
            'num_faces': num_faces,
            'num_edges': num_edges,
            'num_vertices': num_vertices
        }
        
        self.analysis_results['topology'] = topology
        
        print(f"  Sólidos (SOLID):     {num_solids:>8,}")
        print(f"  Shells:              {num_shells:>8,}")
        print(f"  Caras (FACE):        {num_faces:>8,}")
        print(f"  Aristas (EDGE):      {num_edges:>8,}")
        print(f"  Vértices (VERTEX):   {num_vertices:>8,}")
        
        # Evaluación
        print(f"\n  💡 Evaluación:")
        if num_solids > 0:
            print(f"     ✓ Modelo con sólidos (geometría paramétrica)")
        elif num_faces > 10000:
            print(f"     ⚠️  Modelo triangulado (muchas caras, posible STL)")
        else:
            print(f"     ⚠️  Modelo sin sólidos (solo superficies)")
        
        if num_faces > 100000:
            print(f"     ⚠️  ADVERTENCIA: Modelo muy pesado ({num_faces:,} caras)")
            print(f"        El procesamiento puede ser lento")
        
        print()
        return topology
    
    def analyze_face_types(self):
        """Analiza los tipos de superficies geométricas"""
        if self.shape is None:
            return None
        
        print("🎨 ANÁLISIS DE TIPOS DE SUPERFICIES")
        print("─" * 75)
        
        face_types = defaultdict(int)
        face_areas = defaultdict(list)
        
        explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        
        while explorer.More():
            face = explorer.Current()
            
            try:
                surf_adaptor = BRepAdaptor_Surface(face)
                surf_type = surf_adaptor.GetType()
                
                # Clasificar tipo
                type_name = self._get_surface_type_name(surf_type)
                face_types[type_name] += 1
                
                # Calcular área aproximada
                try:
                    u_min, u_max, v_min, v_max = breptools_UVBounds(face)
                    area = abs((u_max - u_min) * (v_max - v_min))
                    face_areas[type_name].append(area)
                except:
                    pass
                
            except Exception as e:
                face_types['Error/Unknown'] += 1
            
            explorer.Next()
        
        self.analysis_results['face_types'] = dict(face_types)
        
        # Mostrar resultados
        total_faces = sum(face_types.values())
        
        print(f"  {'Tipo de Superficie':<25} {'Cantidad':>10} {'%':>8} {'Área Prom.':>12}")
        print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*12}")
        
        for face_type in sorted(face_types.keys(), key=lambda x: face_types[x], reverse=True):
            count = face_types[face_type]
            percentage = (count / total_faces * 100) if total_faces > 0 else 0
            
            if face_type in face_areas and len(face_areas[face_type]) > 0:
                avg_area = np.mean(face_areas[face_type])
                area_str = f"{avg_area:>10.2f}"
            else:
                area_str = "N/A"
            
            print(f"  {face_type:<25} {count:>10,} {percentage:>7.1f}% {area_str:>12}")
        
        print(f"\n  💡 Interpretación:")
        
        if face_types.get('Plane', 0) / total_faces > 0.8:
            print(f"     ⚠️  Modelo altamente triangulado (>80% caras planas)")
            print(f"        Probable origen: STL o mesh")
        
        if face_types.get('Cylinder', 0) > 0:
            print(f"     ✓ Contiene cilindros (posibles heat stakes)")
        
        if face_types.get('BSpline', 0) > 0 or face_types.get('Bezier', 0) > 0:
            print(f"     ✓ Contiene superficies complejas (CAD nativo)")
        
        print()
        return face_types
    
    def analyze_bounding_box(self):
        """Analiza el bounding box del modelo completo"""
        if self.shape is None:
            return None
        
        print("📦 ANÁLISIS DE DIMENSIONES (BOUNDING BOX)")
        print("─" * 75)
        
        bbox = Bnd_Box()
        brepbndlib_Add(self.shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin
        
        dimensions = {
            'xmin': xmin, 'xmax': xmax, 'size_x': size_x,
            'ymin': ymin, 'ymax': ymax, 'size_y': size_y,
            'zmin': zmin, 'zmax': zmax, 'size_z': size_z
        }
        
        self.analysis_results['dimensions'] = dimensions
        
        print(f"  Eje X: [{xmin:>10.2f}, {xmax:>10.2f}] → Tamaño: {size_x:>10.2f} mm")
        print(f"  Eje Y: [{ymin:>10.2f}, {ymax:>10.2f}] → Tamaño: {size_y:>10.2f} mm")
        print(f"  Eje Z: [{zmin:>10.2f}, {zmax:>10.2f}] → Tamaño: {size_z:>10.2f} mm")
        print(f"\n  Volumen del bounding box: {size_x * size_y * size_z:,.2f} mm³")
        
        print()
        return dimensions
    
    def analyze_physical_properties(self):
        """Analiza propiedades físicas (volumen, superficie, centro de gravedad)"""
        if self.shape is None:
            return None
        
        print("⚖️  PROPIEDADES FÍSICAS")
        print("─" * 75)
        
        # Volumen
        try:
            vol_props = GProp_GProps()
            brepgprop_VolumeProperties(self.shape, vol_props)
            volume = vol_props.Mass()
            cog = vol_props.CentreOfMass()
            
            print(f"  Volumen total:        {volume:,.2f} mm³")
            print(f"  Centro de gravedad:   ({cog.X():.2f}, {cog.Y():.2f}, {cog.Z():.2f})")
            
            self.analysis_results['volume'] = volume
            self.analysis_results['center_of_gravity'] = (cog.X(), cog.Y(), cog.Z())
            
            if volume < 0:
                print(f"     ⚠️  Volumen negativo: normales invertidas")
        except Exception as e:
            print(f"  ⚠️  No se pudo calcular volumen (modelo sin sólidos)")
            self.analysis_results['volume'] = None
        
        # Superficie
        try:
            surf_props = GProp_GProps()
            brepgprop_SurfaceProperties(self.shape, surf_props)
            surface = surf_props.Mass()
            
            print(f"  Superficie total:     {surface:,.2f} mm²")
            self.analysis_results['surface'] = surface
        except Exception as e:
            print(f"  ⚠️  No se pudo calcular superficie")
            self.analysis_results['surface'] = None
        
        print()
        return self.analysis_results
    
    def analyze_sample_faces(self, num_samples=10):
        """Analiza una muestra de caras en detalle"""
        if self.shape is None:
            return None
        
        print(f"🔬 ANÁLISIS DETALLADO DE MUESTRA ({num_samples} caras)")
        print("─" * 75)
        
        explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        sample_count = 0
        
        samples = []
        
        while explorer.More() and sample_count < num_samples:
            face = explorer.Current()
            
            # Centro
            center = self._get_face_center(face)
            
            # Normal
            normal = self._get_face_normal(face)
            
            # Tipo
            try:
                surf_adaptor = BRepAdaptor_Surface(face)
                surf_type = self._get_surface_type_name(surf_adaptor.GetType())
            except:
                surf_type = "Unknown"
            
            # Número de vértices
            vertex_count = 0
            vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex_count += 1
                vertex_explorer.Next()
            
            sample_info = {
                'id': sample_count + 1,
                'type': surf_type,
                'center': center,
                'normal': normal,
                'vertices': vertex_count
            }
            
            samples.append(sample_info)
            
            print(f"\n  Cara #{sample_count + 1}:")
            print(f"    Tipo:       {surf_type}")
            print(f"    Centro:     ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            print(f"    Normal:     ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
            print(f"    Vértices:   {vertex_count}")
            
            sample_count += 1
            explorer.Next()
        
        print()
        return samples
    
    def generate_summary(self):
        """Genera un resumen completo del análisis"""
        print("\n" + "="*75)
        print("📋 RESUMEN DEL ANÁLISIS")
        print("="*75)
        
        if 'topology' in self.analysis_results:
            topo = self.analysis_results['topology']
            print(f"\n🔢 Topología:")
            print(f"   Sólidos:   {topo['num_solids']:>8,}")
            print(f"   Caras:     {topo['num_faces']:>8,}")
            print(f"   Aristas:   {topo['num_edges']:>8,}")
            print(f"   Vértices:  {topo['num_vertices']:>8,}")
        
        if 'dimensions' in self.analysis_results:
            dims = self.analysis_results['dimensions']
            print(f"\n📐 Dimensiones:")
            print(f"   {dims['size_x']:.1f} × {dims['size_y']:.1f} × {dims['size_z']:.1f} mm")
        
        if 'volume' in self.analysis_results and self.analysis_results['volume']:
            print(f"\n⚖️  Propiedades:")
            print(f"   Volumen:    {self.analysis_results['volume']:,.1f} mm³")
            if 'surface' in self.analysis_results:
                print(f"   Superficie: {self.analysis_results['surface']:,.1f} mm²")
        
        if 'face_types' in self.analysis_results:
            print(f"\n🎨 Tipos de Caras:")
            for face_type, count in sorted(self.analysis_results['face_types'].items(), 
                                          key=lambda x: x[1], reverse=True)[:3]:
                print(f"   {face_type}: {count:,}")
        
        print("\n" + "="*75)
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES PARA DETECCIÓN DE HEAT STAKES:\n")
        
        if self.analysis_results.get('topology', {}).get('num_solids', 0) > 0:
            print("✅ El modelo contiene sólidos → Usar detector basado en sólidos")
        else:
            print("⚠️  El modelo NO contiene sólidos → Usar detector basado en clustering de caras")
        
        num_faces = self.analysis_results.get('topology', {}).get('num_faces', 0)
        if num_faces > 50000:
            print("⚠️  Modelo muy pesado → Considerar optimización o simplificación")
            print(f"   Sugerencia: Ajustar parámetros de clustering (eps más grande)")
        
        face_types = self.analysis_results.get('face_types', {})
        if face_types.get('Plane', 0) / num_faces > 0.8:
            print("⚠️  Modelo altamente triangulado → Usar clustering espacial")
        
        if face_types.get('Cylinder', 0) > 0:
            print(f"✅ Detectados {face_types['Cylinder']} cilindros → Buena señal para heat stakes")
        
        print("\n" + "="*75 + "\n")
    
    # ==================== MÉTODOS AUXILIARES ====================
    
    def _get_surface_type_name(self, geom_type):
        """Convierte tipo de geometría a nombre legible"""
        type_map = {
            GeomAbs_Plane: 'Plane',
            GeomAbs_Cylinder: 'Cylinder',
            GeomAbs_Cone: 'Cone',
            GeomAbs_Sphere: 'Sphere',
            GeomAbs_Torus: 'Torus',
            GeomAbs_BezierSurface: 'Bezier',
            GeomAbs_BSplineSurface: 'BSpline',
            GeomAbs_SurfaceOfRevolution: 'Revolution',
            GeomAbs_SurfaceOfExtrusion: 'Extrusion',
            GeomAbs_OffsetSurface: 'Offset',
            GeomAbs_OtherSurface: 'Other'
        }
        return type_map.get(geom_type, 'Unknown')
    
    def _get_face_center(self, face):
        """Calcula centro de una cara"""
        vertices = []
        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            vertex_shape = topods.Vertex(vertex)
            pnt = BRep_Tool.Pnt(vertex_shape)
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
            vertex_explorer.Next()
        
        if len(vertices) > 0:
            return np.mean(vertices, axis=0).tolist()
        return [0, 0, 0]
    
    def _get_face_normal(self, face):
        """Calcula normal de una cara"""
        try:
            surf = BRepAdaptor_Surface(face)
            u_min, u_max, v_min, v_max = breptools_UVBounds(face)
            
            u_mid = (u_min + u_max) / 2
            v_mid = (v_min + v_max) / 2
            
            pnt = gp_Pnt()
            vec_u = gp_Vec()
            vec_v = gp_Vec()
            
            surf.D1(u_mid, v_mid, pnt, vec_u, vec_v)
            normal = vec_u.Crossed(vec_v)
            
            if normal.Magnitude() > 0:
                normal.Normalize()
                return [normal.X(), normal.Y(), normal.Z()]
        except:
            pass
        return [0, 0, 1]


def main():
    """Función principal"""
    
    # Archivo STEP a analizar
    step_file = "heatstake.step"  # Cambiar por tu archivo
    
    print("\n" + "🔬 INICIANDO ANÁLISIS COMPLETO DEL ARCHIVO STEP...")
    
    # Crear analizador
    analyzer = STEPFileAnalyzer(step_file)
    
    # Cargar archivo
    if not analyzer.load_step_file():
        return None
    
    # Ejecutar análisis completo
    analyzer.analyze_topology()
    analyzer.analyze_bounding_box()
    analyzer.analyze_physical_properties()
    analyzer.analyze_face_types()
    analyzer.analyze_sample_faces(num_samples=5)
    
    # Generar resumen
    analyzer.generate_summary()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()