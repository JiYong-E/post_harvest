from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import base64
from pathlib import Path
import logging
import subprocess
import pandas as pd
import shutil
import tempfile
from datetime import datetime
from utils.browning_final import combined_browning_mask
from utils.apple import detect_apple_with_lenticels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="수확의 정석")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# 품질 분류 관련 경로
MELON_SORT_DIR = MODELS_DIR / "melon_sort"
MELON_OUTPUT_DIR = MELON_SORT_DIR / "melon"
SWEET_MODEL_PATH = MELON_SORT_DIR / "best_abs_sweet_model_min.joblib"
SHAPE_MODEL_PATH = MELON_SORT_DIR / "참외_shape_model_2class_color_new.pt"
FIND_CHILD_SCRIPT = MELON_SORT_DIR / "find_child_webp.py"
PIPELINE_SCRIPT = MELON_SORT_DIR / "run_quality_pipeline_flat.py"

models_cache = {}

data_storage = {
    'mask': [],
    'quality': [],
    'extraction': []
}

def load_models(fruit_type: str):
    if fruit_type in models_cache:
        return models_cache[fruit_type]

    model_paths = {
        'melon': MODELS_DIR / 'melon',
        'mandarin': MODELS_DIR / 'mandarin',
        'apple': MODELS_DIR / 'apple'
    }

    if fruit_type not in model_paths:
        raise ValueError(f"지원하지 않는 과일 타입: {fruit_type}")

    model_dir = model_paths[fruit_type]

    yolo_path = model_dir / f'yolo_{fruit_type}.pt'
    sam_encoder_path = model_dir / 'sam_vit_b_01ec64.pth'
    sam_decoder_path = model_dir / f'sam_mask_decoder_{fruit_type}_final.pth'

    if not all([yolo_path.exists(), sam_encoder_path.exists(), sam_decoder_path.exists()]):
        raise FileNotFoundError(f"{fruit_type} 모델 파일이 존재하지 않습니다.")

    yolo_model = YOLO(str(yolo_path))

    sam = sam_model_registry["vit_b"](checkpoint=str(sam_encoder_path))
    decoder_state = torch.load(str(sam_decoder_path), map_location='cpu')
    sam.mask_decoder.load_state_dict(decoder_state)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)
    sam.eval()

    predictor = SamPredictor(sam)

    models_cache[fruit_type] = {
        'yolo': yolo_model,
        'sam_predictor': predictor
    }

    return models_cache[fruit_type]


def process_mask(image_bytes: bytes, models: dict):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("이미지 디코딩 실패")
            return None, None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolo_model = models['yolo']
        sam_predictor = models['sam_predictor']

        results = yolo_model(image_rgb, conf=0.25, verbose=False)

        if len(results[0].boxes) == 0:
            logger.warning("YOLO가 객체를 감지하지 못했습니다")
            return None, None

        logger.info(f"YOLO가 {len(results[0].boxes)}개 객체 감지")

        sam_predictor.set_image(image_rgb)

        final_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        for idx, box in enumerate(results[0].boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            logger.info(f"객체 {idx + 1} 신뢰도: {conf:.3f}, 박스: {xyxy}")

            masks, scores, _ = sam_predictor.predict(box=xyxy, multimask_output=True)

            best_mask = masks[np.argmax(scores)]
            final_mask = np.logical_or(final_mask, best_mask).astype(np.uint8)

        masked_image = image_rgb.copy()
        masked_image[final_mask == 0] = [255, 255, 255]

        return masked_image, final_mask

    except Exception as e:
        logger.error(f"이미지 처리 중 오류: {str(e)}")
        return None, None


def process_melon_mask_for_quality(image_path: str, output_dir: Path, base_name: str):
    try:
        models = load_models('melon')
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolo_model = models['yolo']
        results = yolo_model(image_rgb, conf=0.25, verbose=False)

        if len(results[0].boxes) == 0:
            logger.warning("참외를 감지하지 못했습니다")
            return None

        sam_predictor = models['sam_predictor']
        sam_predictor.set_image(image_rgb)

        box = results[0].boxes.xyxy[0].cpu().numpy()
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)

        mask = masks[np.argmax(scores)]

        full_webp_dir = output_dir / "full_webp"
        full_webp_dir.mkdir(parents=True, exist_ok=True)
        webp_path = full_webp_dir / f"{base_name}_0.webp"
        cv2.imwrite(str(webp_path), image, [cv2.IMWRITE_WEBP_QUALITY, 90])

        full_mask_dir = output_dir / "full_mask"
        full_mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = full_mask_dir / f"{base_name}_0.png"
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_uint8)

        logger.info(f"마스크 생성 완료: {base_name}_0.webp")

        return {
            "webp_path": str(webp_path),
            "mask_path": str(mask_path),
            "filename": f"{base_name}_0.webp"
        }

    except Exception as e:
        logger.error(f"마스크 생성 오류: {str(e)}", exc_info=True)
        return None


def encode_image(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/quality", response_class=HTMLResponse)
async def quality(request: Request):
    return templates.TemplateResponse("quality.html", {"request": request})

@app.get("/extraction", response_class=HTMLResponse)
async def extraction(request: Request):
    return templates.TemplateResponse("extraction.html", {"request": request})

@app.get("/data", response_class=HTMLResponse)
async def data(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})

@app.post("/api/mask/run")
async def run_mask(files: List[UploadFile] = File(...), fruit_type: str = Form(...)):
    try:
        logger.info(f"마스킹 시작 - 과일 타입: {fruit_type}, 파일 수: {len(files)}")

        models = load_models(fruit_type)
        results = []
        failed_files = []

        fruit_name_map = {
            'melon': '참외',
            'apple': '사과',
            'mandarin': '귤'
        }

        for file in files:
            image_bytes = await file.read()
            logger.info(f"처리 중: {file.filename} ({len(image_bytes)} bytes)")

            masked_image, _ = process_mask(image_bytes, models)

            if masked_image is not None:
                results.append({'filename': file.filename, 'masked_image': encode_image(masked_image)})
                logger.info(f"성공: {file.filename}")

                # 데이터 저장
                data_storage['mask'].append({
                    '품종': fruit_name_map.get(fruit_type, fruit_type),
                    '파일명': file.filename,
                    '정확도': '95%',
                    '처리시간': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                failed_files.append(file.filename)
                logger.warning(f"실패: {file.filename}")

        if len(results) == 0:
            return JSONResponse(status_code=400, content={'success': False, 'message': '모든 이미지에서 과일을 감지하지 못했습니다.',
                                                          'failed_files': failed_files})

        message = f'{len(results)}개 이미지 처리 완료'
        if failed_files:
            message += f' ({len(failed_files)}개 실패)'

        return JSONResponse({'success': True, 'results': results, 'message': message, 'failed_files': failed_files})

    except Exception as e:
        logger.error(f"마스킹 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quality/run")
async def run_quality(images: List[UploadFile] = File(...), csv: UploadFile = File(...)):
    temp_dir = None

    try:
        logger.info(f"품질 분류 시작 - 이미지 수: {len(images)}")

        temp_dir = Path(tempfile.mkdtemp(prefix="quality_"))
        logger.info(f"임시 디렉토리: {temp_dir}")

        csv_path = temp_dir / "input.csv"
        csv_content = await csv.read()
        with open(csv_path, "wb") as f:
            f.write(csv_content)

        logger.info(f"CSV 파일 저장: {csv_path}")

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='cp949')

        required_cols = ['storage_day', 'hwabang', 'L_mean', 'A_mean', 'B_mean']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"CSV 파일에 필수 컬럼이 없습니다: {', '.join(missing_cols)}"
            )

        logger.info(f"CSV 검증 완료 - 행 수: {len(df)}")

        melon_output = MELON_OUTPUT_DIR
        melon_output.mkdir(parents=True, exist_ok=True)

        full_webp_dir = melon_output / "full_webp"
        full_mask_dir = melon_output / "full_mask"

        if full_webp_dir.exists():
            shutil.rmtree(full_webp_dir)
        if full_mask_dir.exists():
            shutil.rmtree(full_mask_dir)

        full_webp_dir.mkdir(parents=True, exist_ok=True)
        full_mask_dir.mkdir(parents=True, exist_ok=True)

        processed_files = []

        for idx, image in enumerate(images):
            image_bytes = await image.read()
            temp_image_path = temp_dir / f"temp_{idx}_{image.filename}"

            with open(temp_image_path, "wb") as f:
                f.write(image_bytes)

            logger.info(f"이미지 처리 중 ({idx + 1}/{len(images)}): {image.filename}")

            base_name = Path(image.filename).stem

            result = process_melon_mask_for_quality(
                str(temp_image_path),
                melon_output,
                f"{base_name}"
            )

            if result:
                processed_files.append(result)
                logger.info(f"마스크 생성 성공: {result['filename']}")
            else:
                logger.warning(f"마스크 생성 실패: {image.filename}")

        if len(processed_files) == 0:
            raise HTTPException(
                status_code=400,
                detail="모든 이미지에서 참외를 감지하지 못했습니다."
            )

        logger.info(f"마스크 생성 완료 - 성공: {len(processed_files)}개")

        logger.info("find_child_webp.py 실행 중...")

        find_cmd = [
            "python",
            str(FIND_CHILD_SCRIPT),
            str(csv_path),
            "--base", str(melon_output),
            "--col", "cutout_webp_0"
        ]

        result = subprocess.run(
            find_cmd,
            cwd=str(MELON_SORT_DIR),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        if result.returncode != 0:
            logger.error(f"find_child_webp.py 오류: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"find_child_webp.py 실행 실패: {result.stderr}"
            )

        logger.info(f"find_child_webp.py 완료:\n{result.stdout}")

        logger.info("품질 분류 파이프라인 실행 중...")

        output_csv_path = temp_dir / "output.csv"

        pipeline_cmd = [
            "python",
            str(PIPELINE_SCRIPT),
            "--csv", str(csv_path),
            "--out", str(output_csv_path),
            "--sweet_model", str(SWEET_MODEL_PATH),
            "--shape_model_pt", str(SHAPE_MODEL_PATH),
            "--img_col", "cutout_webp_0",
            "--flat_root", str(melon_output),
            "--use_mask"
        ]

        result = subprocess.run(
            pipeline_cmd,
            cwd=str(MELON_SORT_DIR),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        if result.returncode != 0:
            logger.error(f"품질 분류 파이프라인 오류: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"품질 분류 파이프라인 실행 실패: {result.stderr}"
            )

        logger.info(f"품질 분류 파이프라인 완료:\n{result.stdout}")

        try:
            result_df = pd.read_csv(output_csv_path, encoding='utf-8-sig')
        except:
            result_df = pd.read_csv(output_csv_path, encoding='cp949')

        logger.info(f"결과 CSV 읽기 완료 - 행 수: {len(result_df)}")

        results = []

        for idx, row in result_df.iterrows():
            if idx >= len(processed_files):
                break

            processed = processed_files[idx]

            quality_grade = str(row.get('shape_pred', '미분류'))
            triage = str(row.get('triage_final', ''))

            if triage != 'PASS':
                quality_grade = '미분류'

            img_path = processed['webp_path']
            img = cv2.imread(img_path)

            if img is None:
                logger.warning(f"이미지 로드 실패: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_with_text = img_rgb.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Grade: {quality_grade}"

            (text_width, text_height), baseline = cv2.getTextSize(text, font, 1.2, 2)

            padding = 10
            cv2.rectangle(
                img_with_text,
                (padding, padding),
                (padding + text_width + padding, padding + text_height + padding),
                (255, 255, 255),
                -1
            )

            cv2.rectangle(
                img_with_text,
                (padding, padding),
                (padding + text_width + padding, padding + text_height + padding),
                (200, 200, 200),
                2
            )

            if quality_grade == "특":
                color = (34, 139, 34)
            elif quality_grade == "상":
                color = (30, 144, 255)
            else:
                color = (220, 20, 60)

            cv2.putText(
                img_with_text,
                text,
                (padding + padding // 2, padding + text_height + padding // 2),
                font,
                1.2,
                color,
                2,
                cv2.LINE_AA
            )

            results.append({
                "filename": Path(processed['filename']).name,
                "quality_grade": quality_grade,
                "quality_image": encode_image(img_with_text),
                "sweet_pred": float(row.get('sweet_pred_final', 0)),
                "shape_conf": float(row.get('shape_conf', 0)) if pd.notna(row.get('shape_conf')) else 0.0,
                "triage": triage
            })

            data_storage['quality'].append({
                '품종': '참외',
                '파일명': Path(processed['filename']).name,
                '등급': quality_grade,
                '처리시간': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            logger.info(f"결과 생성 ({idx + 1}): {quality_grade}")

        logger.info(f"품질 분류 완료 - 총 {len(results)}개")

        return JSONResponse({
            "success": True,
            "message": f"{len(results)}개 이미지 품질 분류 완료",
            "results": results
        })

    except subprocess.CalledProcessError as e:
        logger.error(f"서브프로세스 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"품질 분류 파이프라인 실행 중 오류: {str(e)}"
        )
    except Exception as e:
        logger.error(f"품질 분류 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"품질 분류 중 오류 발생: {str(e)}"
        )
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"임시 디렉토리 삭제: {temp_dir}")
            except Exception as e:
                logger.warning(f"임시 디렉토리 삭제 실패: {str(e)}")

@app.post("/api/extraction/run")
async def run_extraction(files: List[UploadFile] = File(...), fruit_type: str = Form(...)):
    try:
        logger.info(f"특징 추출 시작 - 과일 타입: {fruit_type}, 파일 수: {len(files)}")

        results = []
        failed_files = []

        for file in files:
            image_bytes = await file.read()
            logger.info(f"처리 중: {file.filename} ({len(image_bytes)} bytes)")

            if fruit_type == "melon_browning" and file.filename.endswith('.webp'):
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(delete=False, suffix='.webp') as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_path = tmp_file.name

                try:
                    rgb, browning_mask, mask_alpha = combined_browning_mask(tmp_path)

                    masked_image = rgb.copy()
                    masked_image[mask_alpha == 0] = [255, 255, 255]

                    browning_result = rgb.copy()
                    browning_result[browning_mask == 0] = [255, 255, 255]

                    # 갈변 비율 계산
                    browning_ratio = (np.sum(browning_mask > 0) / np.sum(mask_alpha > 0)) * 100

                    results.append({
                        'filename': file.filename,
                        'detection_image': encode_image(masked_image),
                        'extraction_image': encode_image(browning_result)
                    })

                    # 데이터 저장
                    data_storage['extraction'].append({
                        '특징조건': '참외 갈변',
                        '파일명': file.filename,
                        '비율': f'{browning_ratio:.1f}%',
                        '처리시간': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                    logger.info(f"성공: {file.filename}")
                except Exception as e:
                    logger.error(f"갈변 추출 오류: {str(e)}")
                    failed_files.append(file.filename)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            elif fruit_type == "apple_lenticel":
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_path = tmp_file.name

                with tempfile.TemporaryDirectory() as tmp_output_dir:
                    try:
                        result = detect_apple_with_lenticels(
                            img_path=tmp_path,
                            model_path=str(MODELS_DIR / 'apple'),
                            output_dir=tmp_output_dir,
                            K_lenticel=4,
                            lenticel_h_ranges=((0, 180),),
                            lenticel_v_min=180,
                            lenticel_s_max=100,
                            local_ksize=51,
                            local_diff_thresh=12,
                            refine_ksize=61,
                            refine_diff_thresh=6,
                            morph_open_kernel=(5, 5),
                            morph_close_kernel=(3, 3),
                        )

                        if result:
                            visualization_path = os.path.join(tmp_output_dir, 'apple_lenticels_visualization.jpg')
                            mask_path = os.path.join(tmp_output_dir, 'apple_lenticels_mask.jpg')

                            viz_img = cv2.imread(visualization_path)
                            viz_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)

                            mask_img = cv2.imread(mask_path)
                            mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

                            results.append({
                                'filename': file.filename,
                                'detection_image': encode_image(viz_rgb),
                                'extraction_image': encode_image(mask_rgb)
                            })

                            # 데이터 저장
                            data_storage['extraction'].append({
                                '특징조건': '사과 과점',
                                '파일명': file.filename,
                                '비율': f"{result['ratio'] * 100:.1f}%",
                                '처리시간': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })

                            logger.info(f"성공: {file.filename} - 과점 비율: {result['ratio']:.4f}")
                        else:
                            failed_files.append(file.filename)
                            logger.warning(f"과점 검출 실패: {file.filename}")

                    except Exception as e:
                        logger.error(f"과점 검출 오류: {str(e)}", exc_info=True)
                        failed_files.append(file.filename)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
            else:
                models = load_models(fruit_type)
                masked_image, final_mask = process_mask(image_bytes, models)

                if masked_image is not None and final_mask is not None:
                    results.append({'filename': file.filename, 'extracted_image': encode_image(masked_image)})
                    logger.info(f"성공: {file.filename}")
                else:
                    failed_files.append(file.filename)
                    logger.warning(f"실패: {file.filename}")

        if len(results) == 0:
            return JSONResponse(status_code=400, content={'success': False, 'message': '모든 이미지에서 과일 특징을 추출하지 못했습니다.',
                                                          'failed_files': failed_files})

        message = f'{len(results)}개 이미지 처리 완료'
        if failed_files:
            message += f' ({len(failed_files)}개 실패)'

        return JSONResponse({'success': True, 'results': results, 'message': message, 'failed_files': failed_files})

    except Exception as e:
        logger.error(f"특징 추출 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{data_type}")
async def get_data(data_type: str):
    try:
        logger.info(f"데이터 조회 요청 - 타입: {data_type}")

        if data_type not in ['mask', 'quality', 'extraction']:
            raise HTTPException(status_code=400, detail="지원하지 않는 데이터 타입입니다.")

        data = data_storage.get(data_type, [])

        return JSONResponse({
            "success": True,
            "data": data,
            "message": f"{data_type} 데이터 조회 완료"
        })

    except Exception as e:
        logger.error(f"데이터 조회 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/program")
async def download_program():
    zip_path = BASE_DIR / "static" / "downloads" / "publish.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="프로그램 파일을 찾을 수 없습니다.")
    return FileResponse(path=str(zip_path), filename="publish.zip", media_type="application/zip")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)