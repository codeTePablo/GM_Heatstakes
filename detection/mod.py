"""
DETECTOR DE HEAT STAKES - Clustering de Cilindros (Optimizado para piezas ensambladas)
Detecta heat stakes en archivos STEP usando clustering de cilindros.

------------------------------------------------------------
Developer: Pablo
Contacto:  codewithpablo@gmail.com
------------------------------------------------------------
"""

import sys
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_Shape
from OCC.Extend.DataExchange import read_step_file, write_step_file
from OCC.Display.OCCViewer import Viewer3d
from OCC.Display.SimpleGui import init_display
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import time

class HeatStakeDetector:
    """Detector de heat stakes para piezas ensambladas"""
    
    def __init__(self, step_file):
        self.step_file = step_file
        self.shape = None
        self.cylinders = []
        self.heat_stakes = []
        self.rejected_clusters = []  # Guardar clusters rechazados
        
        # CRITERIO FLEXIBLE: Número de cilindros por heat stake
        self.MIN_CYLINDERS = 5   # Mínimo absoluto
        self.MAX_CYLINDERS = 25  # Máximo absoluto
        self.TARGET_CYLINDERS = 7  # Referencia ideal
        
        # Parámetros de clustering
        self.DEFAULT_EPS = 25.0
        self.DEFAULT_MIN_SAMPLES = 5
        
        # Modo de análisis
        self.STRICT_MODE = False
        
        # FILTROS para evitar falsos positivos (agujeros)
        self.MIN_SPREAD = 8.0   # mm - Heat stakes compactos aceptables
        self.MAX_RADIUS = 5.0   # mm - Cilindros pequeños
        self.MIN_HEIGHT = 8.0   # mm - Altura mínima
        
        # Sistema de scoring
        self.USE_SMART_FILTERING = True
        self.SCORE_THRESHOLD = 0.5  # 50% del score máximo (ajustable: 0.4 para 40%)
        
    def load_step(self):
        """Carga el archivo STEP"""
        print(f"\n📂 Cargando archivo: {self.step_file}")
        reader = STEPControl_Reader()
        status = reader.ReadFile(self.step_file)
        
        if status != 1:
            raise Exception("❌ Error al leer el archivo STEP")
        
        reader.TransferRoots()
        self.shape = reader.OneShape()
        print("✓ Archivo cargado correctamente")
        return self.shape
    
    def extract_cylinders(self):
        """Extrae todas las caras cilíndricas con sus propiedades"""
        print("\n🔍 Extrayendo cilindros del modelo...")
        
        explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        cylinder_count = 0
        
        while explorer.More():
            face = explorer.Current()
            surf = BRepAdaptor_Surface(face)
            
            if surf.GetType() == GeomAbs_Cylinder:
                cylinder_geom = surf.Cylinder()
                axis = cylinder_geom.Axis()
                location = axis.Location()
                direction = axis.Direction()
                radius = cylinder_geom.Radius()
                
                u_min, u_max, v_min, v_max = breptools_UVBounds(face)
                height = abs(v_max - v_min)
                
                props = GProp_GProps()
                brepgprop_SurfaceProperties(face, props)
                area = props.Mass()
                
                cylinder_info = {
                    'face': face,
                    'center': (location.X(), location.Y(), location.Z()),
                    'direction': (direction.X(), direction.Y(), direction.Z()),
                    'radius': radius,
                    'height': height,
                    'area': area
                }
                
                self.cylinders.append(cylinder_info)
                cylinder_count += 1
            
            explorer.Next()
        
        print(f"✓ Encontrados {cylinder_count} cilindros")
        return self.cylinders
    
    def cluster_cylinders(self, eps=None, min_samples=None):
        """Agrupa cilindros cercanos usando DBSCAN"""
        
        if eps is None:
            eps = self.DEFAULT_EPS
        if min_samples is None:
            min_samples = self.DEFAULT_MIN_SAMPLES
            
        print(f"\n🔬 Agrupando cilindros cercanos...")
        print(f"   Parámetros: eps={eps:.1f}mm, min_samples={min_samples}")
        
        if len(self.cylinders) == 0:
            print("⚠️ No hay cilindros para agrupar")
            return [], []
        
        centers = np.array([cyl['center'] for cyl in self.cylinders])
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(centers)
        
        clusters = defaultdict(list)
        noise_count = 0
        
        for idx, label in enumerate(labels):
            if label == -1:
                noise_count += 1
            else:
                clusters[label].append(idx)
        
        n_clusters = len(clusters)
        print(f"✓ Formados {n_clusters} clusters")
        print(f"   Cilindros sin grupo (ruido): {noise_count}")
        
        return clusters, labels
    
    def analyze_cluster(self, cluster_indices):
        """Analiza un cluster para caracterizarlo"""
        
        cluster_cylinders = [self.cylinders[i] for i in cluster_indices]
        num_cylinders = len(cluster_cylinders)
        
        centers = np.array([cyl['center'] for cyl in cluster_cylinders])
        centroid = centers.mean(axis=0)
        
        distances = np.linalg.norm(centers - centroid, axis=1)
        max_spread = distances.max()
        avg_spread = distances.mean()
        
        min_coords = centers.min(axis=0)
        max_coords = centers.max(axis=0)
        dimensions = max_coords - min_coords
        bbox_volume = np.prod(dimensions) if np.all(dimensions > 0) else 0
        
        radii = [cyl['radius'] for cyl in cluster_cylinders]
        avg_radius = np.mean(radii)
        std_radius = np.std(radii)
        
        heights = [cyl['height'] for cyl in cluster_cylinders]
        avg_height = np.mean(heights)
        max_height = np.max(heights)
        
        directions = np.array([cyl['direction'] for cyl in cluster_cylinders])
        direction_std = np.std(directions, axis=0).sum()
        
        analysis = {
            'num_cylinders': num_cylinders,
            'centroid': tuple(centroid),
            'max_spread': max_spread,
            'avg_spread': avg_spread,
            'dimensions': tuple(dimensions),
            'bbox_volume': bbox_volume,
            'avg_radius': avg_radius,
            'std_radius': std_radius,
            'avg_height': avg_height,
            'max_height': max_height,
            'direction_std': direction_std,
            'cylinder_indices': cluster_indices,
            'cylinders': cluster_cylinders
        }
        
        return analysis
    
    def is_heat_stake(self, cluster_analysis):
        """
        Valida si un cluster es un heat stake
        USA SCORING MULTI-CRITERIO para balance entre precisión y recall
        """
        
        num_cyl = cluster_analysis['num_cylinders']
        spread = cluster_analysis['max_spread']
        avg_radius = cluster_analysis['avg_radius']
        avg_height = cluster_analysis['avg_height']
        max_height = cluster_analysis['max_height']
        direction_std = cluster_analysis['direction_std']
        bbox_volume = cluster_analysis['bbox_volume']
        
        # VALIDACIÓN 1: Número de cilindros
        if self.STRICT_MODE:
            cyl_valid = (self.TARGET_CYLINDERS - 2) <= num_cyl <= (self.TARGET_CYLINDERS + 2)
            confidence = 'HIGH' if cyl_valid else 'LOW'
        else:
            if self.MIN_CYLINDERS <= num_cyl <= self.MAX_CYLINDERS:
                cyl_valid = True
                diff = abs(num_cyl - self.TARGET_CYLINDERS)
                if diff <= 2:
                    confidence = 'HIGH'
                elif diff <= 5:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
            else:
                cyl_valid = False
                confidence = 'REJECTED'
        
        # SISTEMA DE SCORING (6 puntos máximo)
        score = 0.0
        max_score = 6.0
        reasons = []
        
        # Criterio 1: Dispersión (1.0 puntos)
        if spread >= 15.0:
            score += 1.0
            spread_valid = True
        elif spread >= 9.0:
            score += 0.7  # Compacto pero aceptable
            spread_valid = True
        elif spread >= 8.0:
            score += 0.3  # Muy compacto
            spread_valid = True
        else:
            spread_valid = False
            reasons.append(f"Extremadamente compacto ({spread:.2f}mm < 8mm)")
        
        # Criterio 2: Radio (1.0 puntos)
        if avg_radius <= 3.0:
            score += 1.0
            radius_valid = True
        elif avg_radius <= self.MAX_RADIUS:
            score += 0.7
            radius_valid = True
        elif avg_radius <= 8.0:
            score += 0.3
            radius_valid = True
        else:
            radius_valid = False
            reasons.append(f"Radio muy grande ({avg_radius:.2f}mm > 8mm)")
        
        # Criterio 3: Altura (1.0 puntos)
        if max_height > 15.0 or avg_height > 10.0:
            score += 1.0
            height_valid = True
        elif max_height > 10.0 or avg_height > self.MIN_HEIGHT:
            score += 0.7
            height_valid = True
        elif max_height > 5.0 or avg_height > 5.0:
            score += 0.3
            height_valid = True
        else:
            height_valid = False
            reasons.append(f"Muy plano ({avg_height:.2f}mm < 5mm)")
        
        # Criterio 4: Orientación (1.0 puntos) - CLAVE para detectar agujeros
        if direction_std > 0.1:
            score += 1.0
            orientation_valid = True
        elif direction_std > 0.05:
            score += 0.7
            orientation_valid = True
        elif direction_std > 0.02:
            score += 0.3
            orientation_valid = True
        else:
            orientation_valid = False
            reasons.append(f"Cilindros paralelos (std={direction_std:.4f} < 0.02)")
        
        # Criterio 5: Densidad (0.5 puntos)
        if bbox_volume > 0:
            density = num_cyl / bbox_volume * 1000
            if density > 0.5:
                score += 0.5
                density_valid = True
            elif density > 0.1:
                score += 0.3
                density_valid = True
            else:
                density_valid = False
        else:
            density_valid = True
            score += 0.3
        
        # Criterio 6: Número ideal de cilindros (0.5 puntos)
        if num_cyl == 7:
            score += 0.5
        elif 5 <= num_cyl <= 9:
            score += 0.3
        
        # DETECCIÓN ESPECÍFICA DE AGUJEROS
        is_likely_hole = (
            (spread < 8.0 and direction_std < 0.02) or
            (spread < 10.0 and direction_std < 0.01 and avg_height < 8.0) or
            (avg_radius > 8.0 and direction_std < 0.05)
        )
        
        # DECISIÓN FINAL basada en scoring
        if self.USE_SMART_FILTERING:
            threshold = max_score * self.SCORE_THRESHOLD
            is_valid = (score >= threshold and cyl_valid and not is_likely_hole)
            
            # Ajustar confianza según score
            if is_valid:
                score_pct = score / max_score
                if score_pct >= 0.85:
                    confidence = 'HIGH'
                elif score_pct >= 0.70:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
        else:
            # Modo clásico (todos los criterios)
            is_valid = (cyl_valid and spread_valid and radius_valid and 
                       height_valid and orientation_valid and density_valid and 
                       not is_likely_hole)
        
        if is_likely_hole:
            confidence = 'REJECTED_HOLE'
            is_valid = False
        
        validation = {
            'is_heat_stake': is_valid,
            'cylinder_check': cyl_valid,
            'spread_check': spread_valid,
            'radius_check': radius_valid,
            'density_check': density_valid,
            'height_check': height_valid,
            'orientation_check': orientation_valid,
            'is_likely_hole': is_likely_hole,
            'confidence': confidence,
            'score': score,
            'max_score': max_score,
            'details': {
                'cylinders': f"{num_cyl} (range: {self.MIN_CYLINDERS}-{self.MAX_CYLINDERS})",
                'spread': f"{spread:.2f}mm (min: {self.MIN_SPREAD}mm)",
                'avg_radius': f"{avg_radius:.2f}mm (max: {self.MAX_RADIUS}mm)",
                'density': f"{num_cyl / max(bbox_volume, 1) * 1000:.3f} cyl/mm³",
                'avg_height': f"{avg_height:.2f}mm (min: {self.MIN_HEIGHT}mm)",
                'orientation_std': f"{direction_std:.4f}"
            },
            'rejection_reasons': reasons
        }
        
        return validation
    
    def detect_heat_stakes(self, eps=None, min_samples=None, show_all_clusters=False):
        """Proceso completo de detección"""
        
        print("\n" + "="*80)
        print("🎯 DETECTOR DE HEAT STAKES - Sistema de Scoring Inteligente")
        print("="*80)
        
        start_time = time.time()
        
        self.load_step()
        self.extract_cylinders()
        
        if len(self.cylinders) == 0:
            print("\n⚠️ No se encontraron cilindros")
            return []
        
        clusters, labels = self.cluster_cylinders(eps=eps, min_samples=min_samples)
        
        if len(clusters) == 0:
            print("\n⚠️ No se formaron clusters")
            return []
        
        print(f"\n📊 Analizando {len(clusters)} clusters...")
        print("="*80)
        
        heat_stake_candidates = []
        rejected_clusters = []
        
        for cluster_id, indices in sorted(clusters.items()):
            analysis = self.analyze_cluster(indices)
            validation = self.is_heat_stake(analysis)
            
            if validation['is_heat_stake']:
                heat_stake_candidates.append({
                    'cluster_id': cluster_id,
                    'analysis': analysis,
                    'validation': validation
                })
                
                print(f"\n✅ Heat Stake #{len(heat_stake_candidates)} (Cluster #{cluster_id + 1}):")
                print(f"   Cilindros: {analysis['num_cylinders']}")
                print(f"   Centroide: ({analysis['centroid'][0]:.2f}, {analysis['centroid'][1]:.2f}, {analysis['centroid'][2]:.2f})")
                print(f"   Dispersión: {analysis['max_spread']:.2f}mm")
                print(f"   Score: {validation['score']:.2f}/{validation['max_score']:.2f} ({validation['score']/validation['max_score']*100:.1f}%)")
                print(f"   Confianza: {validation['confidence']}")
            else:
                rejected_clusters.append({
                    'cluster_id': cluster_id,
                    'analysis': analysis,
                    'validation': validation
                })
        
        # Mostrar rechazados
        if show_all_clusters and rejected_clusters:
            print(f"\n" + "-"*80)
            print(f"❌ Clusters rechazados ({len(rejected_clusters)}):")
            
            holes = [r for r in rejected_clusters if r['validation'].get('is_likely_hole')]
            others = [r for r in rejected_clusters if not r['validation'].get('is_likely_hole')]
            
            if holes:
                print(f"\n   🕳️  Detectados como AGUJEROS ({len(holes)}):")
                for item in holes[:5]:
                    cid = item['cluster_id']
                    val = item['validation']
                    print(f"\n   Cluster #{cid + 1}:")
                    if val.get('rejection_reasons'):
                        print(f"   Razones: {'; '.join(val['rejection_reasons'])}")
            
            if others:
                print(f"\n   ❌ Otros rechazados ({len(others)}):")
                for item in others[:5]:
                    cid = item['cluster_id']
                    val = item['validation']
                    score = val.get('score', 0)
                    max_sc = val.get('max_score', 6)
                    print(f"\n   Cluster #{cid + 1}: Score {score:.2f}/{max_sc:.2f} ({score/max_sc*100:.1f}%)")
        
        # Resumen
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("📋 RESUMEN")
        print("="*80)
        print(f"Total cilindros: {len(self.cylinders)}")
        print(f"Clusters formados: {len(clusters)}")
        print(f"Heat stakes detectados: {len(heat_stake_candidates)}")
        print(f"Threshold de scoring: {self.SCORE_THRESHOLD*100:.0f}%")
        
        if rejected_clusters:
            holes = sum(1 for r in rejected_clusters if r['validation'].get('is_likely_hole'))
            print(f"Rechazados como agujeros: {holes}")
            print(f"Otros rechazados: {len(rejected_clusters) - holes}")
        
        print(f"Tiempo: {elapsed:.2f}s")
        
        if len(heat_stake_candidates) > 0:
            print(f"\n🎯 COORDENADAS DE HEAT STAKES:")
            print("-"*80)
            print(f"{'ID':<5} {'Conf':<10} {'X':>10} {'Y':>10} {'Z':>10} {'Cyl':>5} {'Score':>8}")
            print("-"*80)
            for i, hs in enumerate(heat_stake_candidates):
                c = hs['analysis']['centroid']
                conf = hs['validation']['confidence']
                n_cyl = hs['analysis']['num_cylinders']
                score = hs['validation']['score']
                max_sc = hs['validation']['max_score']
                conf_icon = '🟢' if conf == 'HIGH' else '🟡' if conf == 'MEDIUM' else '🔴'
                print(f"HS-{i+1:<3} {conf_icon} {conf:<8} {c[0]:10.2f} {c[1]:10.2f} {c[2]:10.2f} {n_cyl:5} {score:.1f}/{max_sc:.1f}")
        
        # Nota sobre ajustes
        expected = 29
        detected = len(heat_stake_candidates)
        if detected < expected:
            print(f"\n⚠️  Detectados {detected} vs {expected} esperados")
            print("💡 Para detectar MÁS, reduce el threshold:")
            print("   Edita línea ~51: self.SCORE_THRESHOLD = 0.4  (40% en lugar de 50%)")
        elif detected == expected:
            print(f"\n✅ PERFECTO: {expected} heat stakes detectados!")
        
        print("="*80)
        
        self.heat_stakes = heat_stake_candidates
        self.rejected_clusters = rejected_clusters
        return heat_stake_candidates
    
    def visualize_heat_stakes(self, marker_size=5.0, show_model=True, show_rejected=False):
        """Visualiza con etiquetas 3D"""
        
        if not self.heat_stakes and not (show_rejected and self.rejected_clusters):
            print("\n⚠️ No hay nada para visualizar")
            return
        
        print("\n" + "="*80)
        print("🎨 VISUALIZACIÓN 3D")
        print("="*80)
        
        display, start_display, _, _ = init_display()
        
        if show_model and self.shape:
            ais_shape = AIS_Shape(self.shape)
            display.Context.Display(ais_shape, True)
            display.Context.SetTransparency(ais_shape, 0.7, True)
            display.Context.SetColor(ais_shape, Quantity_Color(0.8, 0.8, 0.8, Quantity_TOC_RGB), True)
        
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                  (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
        
        # Heat stakes válidos
        for i, hs in enumerate(self.heat_stakes):
            centroid = hs['analysis']['centroid']
            confidence = hs['validation']['confidence']
            
            if confidence == 'HIGH':
                radius = marker_size * 1.2
            elif confidence == 'MEDIUM':
                radius = marker_size * 1.0
            else:
                radius = marker_size * 0.8
            
            pnt = gp_Pnt(centroid[0], centroid[1], centroid[2])
            sphere = BRepPrimAPI_MakeSphere(pnt, radius).Shape()
            
            color = colors[i % len(colors)]
            ais_sphere = AIS_Shape(sphere)
            display.Context.Display(ais_sphere, True)
            display.Context.SetColor(ais_sphere, Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB), True)
            
            label = f"HS-{i+1}"
            text_pos = gp_Pnt(centroid[0], centroid[1], centroid[2] + radius * 1.5)
            display.DisplayMessage(text_pos, label, height=radius * 0.8, message_color=(0, 0, 0))
        
        # Rechazados
        if show_rejected and self.rejected_clusters:
            for j, item in enumerate(self.rejected_clusters):
                centroid = item['analysis']['centroid']
                pnt = gp_Pnt(centroid[0], centroid[1], centroid[2])
                radius = marker_size * 0.6
                sphere = BRepPrimAPI_MakeSphere(pnt, radius).Shape()
                
                ais_sphere = AIS_Shape(sphere)
                display.Context.Display(ais_sphere, True)
                display.Context.SetColor(ais_sphere, Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB), True)
                
                label = f"R-{j+1}"
                text_pos = gp_Pnt(centroid[0], centroid[1], centroid[2] + radius * 1.5)
                display.DisplayMessage(text_pos, label, height=radius * 0.8, message_color=(0.3, 0.3, 0.3))
        
        print(f"Visualizando {len(self.heat_stakes)} heat stakes")
        if show_rejected:
            print(f"           + {len(self.rejected_clusters)} rechazados")
        
        display.FitAll()
        start_display()
    
    def export_visualization_script(self, output_file='visualize_results.py', include_rejected=False):
        """Exporta script de visualización"""
        
        if not self.heat_stakes:
            return
        
        print(f"\n💾 Exportando script de visualización a '{output_file}'...")
        
        with open(output_file, 'w') as f:
            f.write('"""\n')
            f.write('Script de visualización de Heat Stakes detectados\n')
            f.write(f'Archivo original: {self.step_file}\n')
            f.write(f'Heat stakes detectados: {len(self.heat_stakes)}\n')
            if include_rejected:
                f.write(f'Clusters rechazados: {len(self.rejected_clusters)}\n')
            f.write('"""\n\n')
            
            # Heat stakes válidos
            f.write('# Coordenadas de heat stakes VÁLIDOS\n')
            f.write('# Formato: (x, y, z, confianza)\n')
            f.write('heat_stakes = [\n')
            for hs in self.heat_stakes:
                c = hs['analysis']['centroid']
                conf = hs['validation']['confidence']
                f.write(f'    ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, "{conf}"),\n')
            f.write(']\n\n')
            
            # Clusters rechazados
            if include_rejected and self.rejected_clusters:
                f.write('# Coordenadas de clusters RECHAZADOS\n')
                f.write('rejected_clusters = [\n')
                for item in self.rejected_clusters:
                    c = item['analysis']['centroid']
                    f.write(f'    ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}),\n')
                f.write(']\n\n')
            else:
                f.write('rejected_clusters = []\n\n')
            
            f.write('# Cargar modelo\n')
            f.write(f'step_file = "{self.step_file}"\n')
            f.write('reader = STEPControl_Reader()\n')
            f.write('reader.ReadFile(step_file)\n')
            f.write('reader.TransferRoots()\n')
            f.write('shape = reader.OneShape()\n\n')
            
            f.write('# Inicializar visualización\n')
            f.write('display, start_display, _, _ = init_display()\n\n')
            
            f.write('# Mostrar modelo\n')
            f.write('ais_shape = AIS_Shape(shape)\n')
            f.write('display.Context.Display(ais_shape, True)\n')
            f.write('display.Context.SetTransparency(ais_shape, 0.7, True)\n')
            f.write('display.Context.SetColor(ais_shape, Quantity_Color(0.8, 0.8, 0.8, Quantity_TOC_RGB), True)\n\n')
            
            f.write('# Colores para heat stakes válidos\n')
            f.write('colors = [\n')
            f.write('    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),\n')
            f.write('    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),\n')
            f.write(']\n\n')
            
            f.write('# Mostrar heat stakes VÁLIDOS (colores)\n')
            f.write('for i, (x, y, z, conf) in enumerate(heat_stakes):\n')
            f.write('    # Tamaño según confianza\n')
            f.write('    if conf == "HIGH":\n')
            f.write('        radius = 8.0 * 1.2\n')
            f.write('    elif conf == "MEDIUM":\n')
            f.write('        radius = 8.0\n')
            f.write('    else:\n')
            f.write('        radius = 8.0 * 0.8\n')
            f.write('    \n')
            f.write('    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), radius).Shape()\n')
            f.write('    ais_sphere = AIS_Shape(sphere)\n')
            f.write('    display.Context.Display(ais_sphere, True)\n')
            f.write('    color = colors[i % len(colors)]\n')
            f.write('    display.Context.SetColor(ais_sphere, Quantity_Color(*color, Quantity_TOC_RGB), True)\n')
            f.write('    \n')
            f.write('    # Agregar etiqueta\n')
            f.write('    label = f"HS-{i+1}"\n')
            f.write('    text_pos = gp_Pnt(x, y, z + radius * 1.5)\n')
            f.write('    display.DisplayMessage(text_pos, label, height=radius * 0.8, message_color=(0, 0, 0))\n\n')
            
            f.write('# Mostrar clusters RECHAZADOS (negro)\n')
            f.write('for j, (x, y, z) in enumerate(rejected_clusters):\n')
            f.write('    radius = 8.0 * 0.6\n')
            f.write('    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), radius).Shape()\n')
            f.write('    ais_sphere = AIS_Shape(sphere)\n')
            f.write('    display.Context.Display(ais_sphere, True)\n')
            f.write('    display.Context.SetColor(ais_sphere, Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB), True)\n')
            f.write('    \n')
            f.write('    # Agregar etiqueta\n')
            f.write('    label = f"R-{j+1}"\n')
            f.write('    text_pos = gp_Pnt(x, y, z + radius * 1.5)\n')
            f.write('    display.DisplayMessage(text_pos, label, height=radius * 0.8, message_color=(0.3, 0.3, 0.3))\n\n')
            
            f.write('print(f"Visualizando {len(heat_stakes)} heat stakes válidos")\n')
            f.write('if rejected_clusters:\n')
            f.write('    print(f"            + {len(rejected_clusters)} clusters rechazados (negro)")\n')
            f.write('display.FitAll()\n')
            f.write('start_display()\n')
        
        print(f"✓ Script creado: {output_file}")
        print(f"   Para visualizar más tarde, ejecuta: python {output_file}")


def main():
    """Función principal"""
    
    if len(sys.argv) < 2:
        print("="*80)
        print("DETECTOR DE HEAT STAKES - Uso")
        print("="*80)
        print("\n📝 Sintaxis:")
        print("   python detector.py <archivo.step> [eps] [min_samples] [--all] [--view] [--show-rejected] [--export-view] [--strict]")
        print("\n📌 Ejemplos:")
        print("   python detector.py modelo.step")
        print("   python detector.py modelo.step --view --show-rejected")
        print("   python detector.py modelo.step 25.0 4 --all --view --show-rejected")
        print("   python detector.py modelo.step --strict")
        print("\n⚙️  Parámetros:")
        print("   eps              Distancia máxima entre cilindros (default: 25.0mm)")
        print("   min_samples      Mínimo cilindros por grupo (default: 5)")
        print("   --all            Mostrar también clusters rechazados en terminal")
        print("   --view           Abrir visualización 3D interactiva")
        print("   --show-rejected  Mostrar clusters rechazados en NEGRO en 3D")
        print("   --export-view    Crear script Python para visualizar después")
        print("   --strict         Modo estricto: solo acepta 7±2 cilindros")
        print("\n🎯 Modos de Detección:")
        print("   FLEXIBLE (default): Acepta 5-25 cilindros, confianza por cercanía a 7")
        print("   ESTRICTO (--strict): Solo acepta 7±2 cilindros")
        print("\n🎨 Visualización 3D:")
        print("   🔴🟢🔵🟡 Heat stakes VÁLIDOS (colores variados, tamaño según confianza)")
        print("   ⚫ Clusters RECHAZADOS (negro, pequeños) [con --show-rejected]")
        print("\n💡 Tips:")
        print("   - Usa --show-rejected para ver QUÉ no fue clasificado como heat stake")
        print("   - Usa --all para ver detalles de clusters rechazados en terminal")
        print("   - Detecta menos? → Aumenta eps (30-40mm) o reduce min_samples (3-4)")
        print("   - Detecta más? → Reduce eps (15-20mm) o aumenta min_samples (6-8)")
        print("="*80)
        sys.exit(1)
    
    step_file = sys.argv[1]
    eps = None
    min_samples = None
    show_all = '--all' in sys.argv
    show_view = '--view' in sys.argv
    show_rejected = '--show-rejected' in sys.argv
    export_view_script = '--export-view' in sys.argv
    strict_mode = '--strict' in sys.argv
    
    # Parse numeric arguments
    for i, arg in enumerate(sys.argv[2:], start=2):
        if not arg.startswith('--'):
            try:
                val = float(arg)
                if eps is None:
                    eps = val
                elif min_samples is None:
                    min_samples = int(val)
            except ValueError:
                pass
    
    # Crear y ejecutar detector
    detector = HeatStakeDetector(step_file)
    
    # Configurar modo estricto si se solicitó
    if strict_mode:
        detector.STRICT_MODE = True
        print("\n⚠️  Modo ESTRICTO activado: Solo 7±2 cilindros\n")
    
    heat_stakes = detector.detect_heat_stakes(
        eps=eps, 
        min_samples=min_samples,
        show_all_clusters=show_all
    )
    
    # Exportar coordenadas
    if heat_stakes:
        output_file = 'heat_stakes_coordinates.txt'
        print(f"\n💾 Exportando a '{output_file}'...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("COORDENADAS DE HEAT STAKES DETECTADOS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total detectados: {len(heat_stakes)}\n")
            f.write(f"Archivo: {step_file}\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("="*80 + "\n")
            f.write("HEAT STAKES VÁLIDOS\n")
            f.write("="*80 + "\n\n")
            
            for i, hs in enumerate(heat_stakes):
                c = hs['analysis']['centroid']
                n = hs['analysis']['num_cylinders']
                s = hs['analysis']['max_spread']
                conf = hs['validation']['confidence']
                
                f.write(f"┌─ Heat Stake HS-{i+1} {'─' * (60 - len(str(i+1)))}\n")
                f.write(f"│  ID:         HS-{i+1}\n")
                f.write(f"│  Centroide:  X={c[0]:.4f}, Y={c[1]:.4f}, Z={c[2]:.4f}\n")
                f.write(f"│  Cilindros:  {n}\n")
                f.write(f"│  Dispersión: {s:.2f} mm\n")
                f.write(f"│  Confianza:  {conf}\n")
                if 'score' in hs['validation']:
                    score = hs['validation']['score']
                    max_sc = hs['validation']['max_score']
                    f.write(f"│  Score:      {score:.2f}/{max_sc:.2f} ({score/max_sc*100:.1f}%)\n")
                f.write(f"└{'─' * 70}\n\n")
            
            # CSV format section para fácil importación
            f.write("\n" + "="*80 + "\n")
            f.write("FORMATO CSV (para importar a Excel/software)\n")
            f.write("="*80 + "\n\n")
            f.write("ID,X,Y,Z,Cilindros,Dispersión,Confianza\n")
            for i, hs in enumerate(heat_stakes):
                c = hs['analysis']['centroid']
                n = hs['analysis']['num_cylinders']
                s = hs['analysis']['max_spread']
                conf = hs['validation']['confidence']
                f.write(f"HS-{i+1},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{n},{s:.2f},{conf}\n")
            
            # Sección de rechazados (si existen)
            if detector.rejected_clusters:
                f.write("\n\n" + "="*80 + "\n")
                f.write("CLUSTERS RECHAZADOS (No clasificados como Heat Stakes)\n")
                f.write("="*80 + "\n\n")
                
                for j, item in enumerate(detector.rejected_clusters):
                    c = item['analysis']['centroid']
                    n = item['analysis']['num_cylinders']
                    s = item['analysis']['max_spread']
                    val = item['validation']
                    
                    f.write(f"┌─ Cluster Rechazado R-{j+1} {'─' * (50 - len(str(j+1)))}\n")
                    f.write(f"│  ID:         R-{j+1}\n")
                    f.write(f"│  Centroide:  X={c[0]:.4f}, Y={c[1]:.4f}, Z={c[2]:.4f}\n")
                    f.write(f"│  Cilindros:  {n}\n")
                    f.write(f"│  Dispersión: {s:.2f} mm\n")
                    
                    # Indicar si es agujero
                    if val.get('is_likely_hole'):
                        f.write(f"│  TIPO: Agujero/Hoyo detectado 🕳️\n")
                    
                    f.write(f"│  Razón de rechazo:\n")
                    f.write(f"│    - Cilindros: {'✓' if val['cylinder_check'] else '✗'} {val['details']['cylinders']}\n")
                    f.write(f"│    - Dispersión: {'✓' if val['spread_check'] else '✗'} {val['details']['spread']}\n")
                    f.write(f"│    - Radio: {'✓' if val['radius_check'] else '✗'} {val['details']['avg_radius']}\n")
                    
                    # Validaciones que pueden no existir en modo estricto
                    if 'height_check' in val:
                        f.write(f"│    - Altura: {'✓' if val['height_check'] else '✗'} {val['details']['avg_height']}\n")
                    if 'orientation_check' in val:
                        f.write(f"│    - Orientación: {'✓' if val['orientation_check'] else '✗'} {val['details']['orientation_std']}\n")
                    if 'density_check' in val:
                        f.write(f"│    - Densidad: {'✓' if val['density_check'] else '✗'} {val['details']['density']}\n")
                    
                    # Razones específicas
                    if val.get('rejection_reasons'):
                        f.write(f"│  Detalles: {'; '.join(val['rejection_reasons'])}\n")
                    
                    f.write(f"└{'─' * 70}\n\n")
                
                # CSV para rechazados
                f.write("\nCSV - RECHAZADOS:\n")
                f.write("ID,X,Y,Z,Cilindros,Dispersión,Razón\n")
                for j, item in enumerate(detector.rejected_clusters):
                    c = item['analysis']['centroid']
                    n = item['analysis']['num_cylinders']
                    s = item['analysis']['max_spread']
                    is_hole = item['validation'].get('is_likely_hole')
                    razon = "Agujero" if is_hole else "Criterios_no_cumplidos"
                    f.write(f"R-{j+1},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{n},{s:.2f},{razon}\n")
        
        print(f"✓ Archivo creado")
        
        # Visualización 3D
        if show_view:
            detector.visualize_heat_stakes(marker_size=8.0, show_model=True)


if __name__ == "__main__":
    main()