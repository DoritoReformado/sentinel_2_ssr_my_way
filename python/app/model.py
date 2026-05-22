import os
import requests
import geopandas as gpd
import pandas as pd
import rasterio
from datetime import datetime
#vamos a obtener el header de credenciales para copernicus
import requests
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from shapely.geometry import box
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from rasterio.transform import from_bounds
import glob
import rasterio
from rasterio.merge import merge
import numpy as np
import os
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
import shutil


for year_date in range(2025, 2026):
    url = "https://stac.dataspace.copernicus.eu/v1/"
    username = os.environ.get("COPERNICUS_USERNAME", "")
    password = os.environ.get("COPERNICUS_PASSWORD", "")
    study_area = gpd.read_file("./data/geojson/silvia_bounding_box.geojson")
    if study_area.crs != "EPSG:4326":
        study_area = study_area.to_crs("EPSG:4326")
    start_date = datetime.strptime(f"{year_date}-01-01", "%Y-%m-%d")
    end_date = datetime.strptime(f"{year_date}-12-31", "%Y-%m-%d")
    cloud_cover = 40
    bounds = study_area.total_bounds
    bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]

    search_url = f"{url}search"
    search_params = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}/{end_date.strftime('%Y-%m-%dT23:59:59Z')}",
        # 'input': ['sentinel-2-l2a'],
        "bbox": bbox,
        "query": {
            "eo:cloud_cover": {
                "lt": cloud_cover
            }
        },
        "sortby": [
            {
                "field": "properties.eo:cloud_cover",
                "direction": "asc"
            }
        ],
        "limit": 100
    }

    response = requests.post(search_url, json=search_params, auth=(username, password))
    respuesta = response.json()
    df_download_data = pd.DataFrame(respuesta["features"])



    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    response = requests.post(auth_url, data=data)
    token = response.json()["access_token"]

    headers = {
        "Authorization": f"Bearer {token}"
    }
    bandas = [
        # 10 m (base para superresolución)
        "B02_10m",  # Blue
        "B03_10m",  # Green
        "B04_10m",  # Red
        "B08_10m",  # NIR

        # 20 m (a superresolver)
        "B05_20m",  # Red-edge
        "B06_20m",
        "B07_20m",
        "B8A_20m",  # NIR narrow
        "B11_20m",  # SWIR
        "B12_20m",

        # auxiliares para nubes
        "B09_60m",  # vapor (opcional)
        
        # máscaras y clasificación
        "SCL_20m",  # Scene Classification (CLAVE para nubes)

        # opcional (mejora detección)
        "AOT_20m",  # aerosoles
        "WVP_20m",  # vapor de agua
    ]
    sentinel_folder = "./data/sentinel_2/"
    os.makedirs(sentinel_folder, exist_ok=True)
    df_download_data["grid_code"] = df_download_data["properties"].apply(lambda x: x.get("grid:code", "unknown"))
    df_download_data["start_datetime"] = pd.to_datetime(df_download_data["properties"].apply(lambda x: x.get("datetime", "unknown")))
    df_download_data["cloud_cover"] = df_download_data["properties"].apply(lambda x: x.get("eo:cloud_cover", 100))
    final_resume = []
    for code in df_download_data["grid_code"].unique():
        code_folder = os.path.join(sentinel_folder, code)
        os.makedirs(code_folder, exist_ok=True)
        df_download_data_code = df_download_data[df_download_data["grid_code"] == code]
        for idx, row in df_download_data_code.iterrows():
            str_date = row["start_datetime"].strftime("%Y-%m-%d")
            code_date_folder = os.path.join(code_folder, str_date)
            os.makedirs(code_date_folder, exist_ok=True)
            assets = row["assets"]
            resume = {
                "grid_code": code,
                "datetime": str_date,
                "cloud_cover": row["cloud_cover"],
                "download_data":{}
            }
            try:
                for asset_key, asset_info in assets.items():
                    if asset_key in bandas:
                        asset_url = asset_info["alternate"]["https"]["href"]
                        if asset_url.startswith("s3://"):
                            asset_url = asset_url.replace(
                                "s3://eodata/",
                                "https://eodata.sentinel-copernicus.eu/"
                            )

                        #asset_url = asset_info["href"]
                        filename = f"{asset_key}.jp2"
                        filepath = os.path.join(code_date_folder, filename)
                        if not os.path.exists(filepath):
                            asset_response = requests.get(asset_url, headers=headers, stream=True)
                            with open(filepath, "wb") as f:
                                for chunk in asset_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            resume["download_data"][asset_key] = filepath
                        else:
                            print(f"{asset_key} para {code} en {str_date} ya existe, saltando descarga.")
                            resume["download_data"][asset_key] = filepath
            except Exception as e:
                print(f"no se pudo descargar {code} en {str_date}: {e}")
                # y vamos a borrar la carpeta para evitar inconsistencias
                shutil.rmtree(code_date_folder)
            final_resume.append(resume)
    final_df = pd.DataFrame(final_resume)
    final_df.to_csv(os.path.join(sentinel_folder, "download_summary.csv"), index=False)

    def align_to_base(src_array, src_profile, base_profile):
        aligned = np.zeros((src_array.shape[0],
                            base_profile["height"],
                            base_profile["width"]),
                            dtype=np.float32)

        for i in range(src_array.shape[0]):
            reproject(
                source=src_array[i],
                destination=aligned[i],
                src_transform=src_profile["transform"],
                src_crs=src_profile["crs"],
                dst_transform=base_profile["transform"],
                dst_crs=base_profile["crs"],
                resampling=Resampling.bilinear
            )

        return aligned

    def align_mask(mask, src_profile, base_profile):
        aligned = np.zeros(
            (base_profile["height"], base_profile["width"]),
            dtype=np.float32
        )

        reproject(
            source=mask,
            destination=aligned,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=base_profile["transform"],
            dst_crs=base_profile["crs"],
            resampling=Resampling.nearest
        )

        return aligned


    DATA_PATH = "./data/sentinel_2/"

    # =========================
    # Cargar área de estudio
    # =========================
    study_area = gpd.read_file("./data/geojson/silvia_bounding_box.geojson")

    study_area_buffer = study_area.to_crs(epsg=3116).buffer(5000)
    study_area_buffer = gpd.GeoDataFrame(geometry=study_area_buffer, crs="EPSG:3116")

    study_area_utm = study_area.to_crs(epsg=3116)  # ajusta zona si es necesario

    minx, miny, maxx, maxy = study_area_utm.total_bounds

    # =========================
    # Crear grilla
    # =========================
    tile_size = 4000  # metros
    grid = []

    x_coords = np.arange(minx, maxx, tile_size)
    y_coords = np.arange(miny, maxy, tile_size)

    for x in x_coords:
        for y in y_coords:
            tile = box(x, y, x + tile_size, y + tile_size)
            grid.append(tile)

    grid_gdf = gpd.GeoDataFrame(geometry=grid, crs=study_area_utm.crs)
    grid_gdf = grid_gdf.to_crs(study_area.crs)


    # =========================
    # Cargar metadata
    # =========================
    df_data = pd.read_csv(os.path.join(DATA_PATH, "download_summary.csv"))
    df_data = df_data.sort_values(by="cloud_cover").reset_index(drop=True)

    # Convertir columna de string a dict
    def ensure_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return None
        return None

    df_data["download_data"] = df_data["download_data"].apply(ensure_dict)

    # =========================
    # Funciones auxiliares
    # =========================
    def get_band_path(bands_dict, band_name):
        key = [k for k in bands_dict if band_name in k][0]
        return bands_dict[key]



    def read_and_align(path, base_profile, geom, resampling):
        with rasterio.open(path) as src:

            # recortar primero (MUY IMPORTANTE)
            geom_proj = geom.to_crs(src.crs)
            out_image, out_transform = rio_mask(src, geom_proj.geometry, crop=True)

            # luego reproyectar SOLO ese recorte
            dst = np.zeros(
                (base_profile["height"], base_profile["width"]),
                dtype=np.float32
            )

            reproject(
                source=out_image[0],
                destination=dst,
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=base_profile["transform"],
                dst_crs=base_profile["crs"],
                resampling=resampling
            )

            return dst

    def load_patch(row, geom, base_profile):
        bands = row["download_data"]

        def read(name, resampling):
            path = get_band_path(bands, name)
            return read_and_align(path, base_profile, geom, resampling)

        # =====================
        # bandas 10m
        # =====================
        b02 = read("B02_10m", Resampling.bilinear)
        b03 = read("B03_10m", Resampling.bilinear)
        b04 = read("B04_10m", Resampling.bilinear)
        b08 = read("B08_10m", Resampling.bilinear)

        # =====================
        # bandas 20m → ya quedan alineadas directo
        # =====================
        b05 = read("B05_20m", Resampling.bilinear)
        b06 = read("B06_20m", Resampling.bilinear)

        input_data = np.stack([b02, b03, b04, b08, b05, b06], axis=0)

        # =====================
        # máscara
        # =====================
        scl = read("SCL_20m", Resampling.nearest)

        nodata_mask = (b02 == 0)
        cloud_mask = np.isin(scl, [3, 8, 9, 10, 11])
        cloud_mask = np.logical_or(cloud_mask, nodata_mask).astype(np.float32)

        return input_data, cloud_mask


    def extract_tiles(img, mask, size=128):
        tiles = []

        for i in range(0, img.shape[1], size):
            for j in range(0, img.shape[2], size):

                x = img[:, i:i + size, j:j + size]
                m = mask[i:i + size, j:j + size]

                if x.shape[1:] == (size, size):
                    tiles.append((x, m))

        return tiles

    def normalize(img):
        return (img - np.mean(img)) / (np.std(img) + 1e-6)

    def cloud_free_composite(images, masks):
        """
        images: lista de np.array (C, H, W)
        masks: lista de np.array (H, W) con 1 = nube, 0 = limpio
        
        retorna: imagen compuesta sin nubes
        """
        
        composite = np.zeros_like(images[0])
        composite_mask = np.ones_like(masks[0])  # todo nube al inicio

        for img, mask in zip(images, masks):
            # donde la base tiene nube y la nueva NO tiene nube → reemplazar
            replace = (composite_mask == 1) & (mask == 0)

            for c in range(composite.shape[0]):
                composite[c][replace] = img[c][replace]

            # actualizar máscara
            composite_mask[replace] = 0

            # early stop si ya no hay nubes
            if np.sum(composite_mask) == 0:
                break

        return composite

    def save_tif(path, array, profile):
        profile = profile.copy()

        profile.update({
            "driver": "GTiff",              # 🔥 CLAVE
            "count": array.shape[0],
            "dtype": "float32",            # 🔥 evita problemas
            "compress": "lzw"
        })

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array)
            

    def save_rgb_preview(composite, path):
        rgb = composite[[2, 1, 0], :, :]  # B04, B03, B02
        rgb = np.transpose(rgb, (1, 2, 0))

        # normalización visual
        rgb = rgb / 10000.0
        rgb = np.clip(rgb, 0, 1)

        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()



    def fill_nans_spatial(img):
        filled = img.copy()

        for c in range(img.shape[0]):
            band = filled[c]

            nan_mask = np.isnan(band)

            if np.all(nan_mask):
                continue

            idx = ndimage.distance_transform_edt(
                nan_mask,
                return_distances=False,
                return_indices=True
            )

            filled[c] = band[tuple(idx)]

        return filled

    def fill_nans_by_tiles(img, tile_size=512):
        out = img.copy()

        for i in range(0, img.shape[1], tile_size):
            for j in range(0, img.shape[2], tile_size):
                tile = out[:, i:i+tile_size, j:j+tile_size]
                out[:, i:i+tile_size, j:j+tile_size] = fill_nans_spatial(tile)

        return out



    def create_base_profile(reference_path, geom):
        with rasterio.open(reference_path) as src:
            geom_proj = geom.to_crs(src.crs)
            minx, miny, maxx, maxy = geom_proj.total_bounds

            res = src.transform.a  # 10m típico

            width = int(np.ceil((maxx - minx) / res))
            height = int(np.ceil((maxy - miny) / res))
            transform = from_bounds(minx, miny, maxx, maxy, width, height)

            profile = src.profile.copy()
            profile.update({
                "height": height,
                "width": width,
                "transform": transform,
                "crs": src.crs
            })

            return profile

    # =========================
    # Prueba de carga
    # =========================

    base_profile = None
    base_transform = None
    base_shape = None

    for code in df_data["grid_code"].unique():
        base_profile = None
        df_data_code = df_data[df_data["grid_code"] == code]

        composite = None
        valid_count = None

        for _, row in tqdm(df_data_code.iterrows(), total=len(df_data_code)):
            try:
                print(f"procesando {code} - {row['datetime']} (cloud cover: {row['cloud_cover']}%)")

                # =====================
                # Crear base grid UNA sola vez
                # =====================
                if base_profile is None:
                    ref_path = get_band_path(row["download_data"], "B02_10m")

                    with rasterio.open(ref_path) as src:
                        bounds = src.bounds
                        crs = src.crs

                    tile_geom = gpd.GeoDataFrame(
                        geometry=[box(*bounds)],
                        crs=crs
                    )

                    base_profile = create_base_profile(ref_path, tile_geom)

                # =====================
                # Cargar datos alineados
                # =====================
                input_data, cloud_mask = load_patch(row, study_area_buffer, base_profile)
                input_data = input_data.astype(np.float32)

                # =====================
                # Aplicar máscara
                # =====================
                pixel_mask = np.expand_dims(cloud_mask, axis=0)
                input_data = np.where(pixel_mask == 1, np.nan, input_data)

                # =====================
                # Inicializar acumuladores
                # =====================
                if composite is None:
                    composite = np.zeros_like(input_data, dtype=np.float32)
                    valid_count = np.zeros_like(input_data, dtype=np.float32)

                # =====================
                # Acumulación eficiente
                # =====================
                valid_pixels = ~np.isnan(input_data)

                composite[valid_pixels] += input_data[valid_pixels]
                valid_count[valid_pixels] += 1

            except Exception as e:
                print(f"Error loading patch: {e}")

        # =====================
        # Finalizar composite
        # =====================
        if composite is not None:
            composite = composite / np.maximum(valid_count, 1e-6)
            composite = fill_nans_by_tiles(composite)

            print(f"Composite shape: {composite.shape}")

            folder_resultado = os.path.join("./data/composite/", code)
            os.makedirs(folder_resultado, exist_ok=True)

            save_tif(
                os.path.join(folder_resultado, f"{code}_composite.tif"),
                composite,
                base_profile
            )

            save_rgb_preview(
                composite,
                os.path.join(folder_resultado, f"{code}_preview.png")
            )


    def fill_nans_spatial(img):
        filled = img.copy()

        for c in range(img.shape[0]):
            band = filled[c]

            nan_mask = np.isnan(band)

            if np.all(nan_mask):
                continue

            idx = ndimage.distance_transform_edt(
                nan_mask,
                return_distances=False,
                return_indices=True
            )

            filled[c] = band[tuple(idx)]

        return filled

    def fill_nans_by_tiles(img, tile_size=512):
        out = img.copy()

        for i in range(0, img.shape[1], tile_size):
            for j in range(0, img.shape[2], tile_size):
                tile = out[:, i:i+tile_size, j:j+tile_size]
                out[:, i:i+tile_size, j:j+tile_size] = fill_nans_spatial(tile)

        return out



    composite_folder = "./data/composite/"

    year = str(start_date.year)
    folder_resultados = os.path.join("./data/", "resultados", year)
    os.makedirs(folder_resultados, exist_ok=True)
    composite_folder = "./data/composite/"
    tiff_files = glob.glob(os.path.join(composite_folder, "**", "*.tif"), recursive=True)
    for fp in tiff_files:
        with rasterio.open(fp) as src:
            print(fp, src.count, src.shape, src.dtypes)
    no_data_values = [-9999, 0]

    src_files = []
    for fp in tiff_files:
        src_files.append(rasterio.open(fp))
    mosaic, out_transform = merge(
        src_files,
        nodata=0,          # 🔥 clave
        method="max"     # o "max" (te explico abajo)
    )
    no_data_values = [-9999, 0]

    mask_nodata = np.isin(mosaic, no_data_values)
    mosaic = np.where(mask_nodata, np.nan, mosaic)

    mosaic_filled = fill_nans_by_tiles(mosaic)
    out_meta = src_files[0].meta.copy()

    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": mosaic.shape[0],   # 🔥 importante
        "dtype": "float32",         # 🔥 consistente con tu pipeline
        "nodata": np.nan            # opcional pero recomendado
    })

    output_path = os.path.join(folder_resultados, f"mosaic_{year}.tif")

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic_filled)
        

    scale = 4
    block_size = 512  # puedes ajustar (256–1024 según RAM)

    year = str(start_date.year)

    with rasterio.open(f"./data/resultados/{year}/mosaic_{year}.tif") as src:

        profile = src.profile.copy()

        new_profile = profile.copy()
        new_profile.update({
            "height": profile["height"] * scale,
            "width": profile["width"] * scale,
            "count": 3,
            "dtype": "uint8",
            "nodata": None   # 🔥 ESTA ES LA CLAVE
        })
        
        ndvi_profile = new_profile.copy()
        ndvi_profile.update({
        "count": 1,
        "dtype": "float32"
        })

        transform = profile["transform"]
        new_transform = transform * transform.scale(
            (profile["width"] / new_profile["width"]),
            (profile["height"] / new_profile["height"])
        )
        new_profile["transform"] = new_transform
        ndvi_path = f"./data/resultados/{year}/ndvi_{year}.tif"
                    

        with rasterio.open(f"./data/resultados/{year}/upscaled_{year}.tif", "w", **new_profile) as dst, \
        rasterio.open(ndvi_path, "w", **ndvi_profile) as dst_ndvi:

            for y in range(0, src.height, block_size):
                for x in range(0, src.width, block_size):

                    window = Window(x, y,
                                    min(block_size, src.width - x),
                                    min(block_size, src.height - y))

                    # =====================
                    # leer bloque
                    # =====================
                    composite = src.read(window=window)
                    nir = composite[3]   # B08
                    red = composite[2]   # B04
                    
                    nir = nir.astype(np.float32) / 10000.0
                    red = red.astype(np.float32) / 10000.0

                    nir_up = cv2.resize(
                        nir,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC
                    )

                    red_up = cv2.resize(
                        red,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC
                    )
                    ndvi = (nir_up - red_up) / (nir_up + red_up + 1e-6)
                    
                    # RGB
                    rgb = composite[[2, 1, 0], :, :]
                    rgb = np.transpose(rgb, (1, 2, 0))

                    # normalizar
                    rgb = rgb.astype(np.float32) / 10000.0
                    rgb = np.clip(rgb, 0, 1)

                    rgb_8bit = (rgb * 255).astype(np.uint8)

                    # =====================
                    # upscale
                    # =====================
                    up = cv2.resize(
                        rgb_8bit,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC
                    )

                    # =====================
                    # sharpening
                    # =====================
                    up_f = up.astype(np.float32) / 255.0
                    blur = cv2.GaussianBlur(up_f, (0, 0), sigmaX=1.2)
                    detail = up_f - blur

                    gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                    edges = cv2.Laplacian(gray, cv2.CV_32F)
                    edges = np.abs(edges)

                    edges = edges / (edges.max() + 1e-6)
                    edges = np.expand_dims(edges, axis=2)

                    sharp_f = up_f + detail * edges * 1.5
                    sharp_f = np.clip(sharp_f, 0, 1)

                    sharp = (sharp_f * 255).astype(np.uint8)

                    # =====================
                    # escribir bloque
                    # =====================
                    out_window = Window(
                        x * scale,
                        y * scale,
                        sharp.shape[1],
                        sharp.shape[0]
                    )

                    dst.write(
                        np.transpose(sharp, (2, 0, 1)),
                        window=out_window
                    )
                    
                    dst_ndvi.write(ndvi[np.newaxis, :, :], window=out_window)

    #eliminar informacion en composite_folder para liberar espacio

    shutil.rmtree(composite_folder)
    shutil.rmtree(sentinel_folder)

    print("✅ Procesamiento por bloques terminado")