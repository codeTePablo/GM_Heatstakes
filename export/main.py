from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Shape, TopoDS_Solid
import os

def cargar_step(path):
    """Carga un archivo STEP y devuelve el shape raíz."""
    print(f"-> Cargando STEP: {path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)

    if status != IFSelect_RetDone:
        raise RuntimeError("Error: No se pudo leer el archivo STEP")

    reader.TransferRoots()
    shape = reader.OneShape()
    return shape

def obtener_solidos(shape):
    """Devuelve una lista de todos los sólidos encontrados en el STEP."""
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solidos = []

    while explorer.More():
        solid = TopoDS_Solid.DownCast(explorer.Current())
        solidos.append(solid)
        explorer.Next()

    return solidos

def exportar_solid(solid, filename):
    """Exporta un sólido específico a un archivo STEP."""
    writer = STEPControl_Writer()
    writer.Transfer(solid, STEPControl_AsIs)
    status = writer.Write(filename)

    if status != IFSelect_RetDone:
        raise RuntimeError("Error al escribir el archivo STEP")

    print(f"\n✔ Sólido exportado correctamente a: {filename}")

# ============ MAIN ============

# Ruta del archivo que quieres leer
archivo_step = "EXERCISE 1.stp"   # <-- cámbialo a tu archivo

# 1. Cargar STEP
shape = cargar_step(archivo_step)

# 2. Extraer sólidos
solidos = obtener_solidos(shape)

print("\n===== SÓLIDOS DETECTADOS =====")
for i, s in enumerate(solidos):
    print(f"Sólido ID {i} → {s}")

if not solidos:
    print("No se encontraron sólidos.")
    exit()

# 3. El usuario elige cuál exportar
idx = int(input("\nIngresa el ID del sólido que quieres exportar: "))

if idx < 0 or idx >= len(solidos):
    raise ValueError("ID inválido")

solid_seleccionado = solidos[idx]

# 4. Exportar
output_path = f"pieza_exportada_{idx}.step"
exportar_solid(solid_seleccionado, output_path)
