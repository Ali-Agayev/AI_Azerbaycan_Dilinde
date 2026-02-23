"""
kaggle_client.py — TAM AVTOMATİK Kaggle API İnteqrasiyası
==========================================================
İş axını:
  1. Video faylı Kaggle Dataset kimi upload edir
  2. Notebook-u (kernel) Kaggle API vasitəsilə işlədır
  3. Kernel tamamlanana qədər status izləyir
  4. Nəticə videonu avtomatik yükləyir
"""

import os
import json
import time
import uuid
import zipfile
import tempfile
import shutil
from pathlib import Path

# ─── İş vəziyyəti (yaddaşda saxlanılır) ─────────────────────────────────────
_is_leyi: dict = {}

# ─── Kaggle SDK LAZY import ──────────────────────────────────────────────────
# kaggle.__init__.py avtomatik authenticate() çağırır.
# Modul-leveldə import etmək Railway-də crash-ə səbəb olur.
# Bunun əvəzinə hər SDK çağırışından əvvəl _kaggle_hazirla() çalışdırırıq.


def _kaggle_hazirla() -> bool:
    """
    Kaggle SDK-sını import etməzdən əvvəl konfiq faylını hazırlayır.
    Railway-də KAGGLE_API_TOKEN + KAGGLE_USERNAME env var-larından oxuyur.
    Qaytarır: True (uğurlu), False (konfiq yoxdur)
    """
    api_token = os.environ.get("KAGGLE_API_TOKEN", "")
    username = os.environ.get("KAGGLE_USERNAME", "")

    if api_token and username:
        # Kaggle SDK hər iki path-i yoxlayır — hər ikisini yazırıq
        for qovluq in [
            Path.home() / ".kaggle",
            Path.home() / ".config" / "kaggle",
        ]:
            qovluq.mkdir(parents=True, exist_ok=True)
            fayl = qovluq / "kaggle.json"
            if not fayl.exists():
                with open(fayl, "w") as f:
                    json.dump({"username": username, "key": api_token}, f)
                try:
                    fayl.chmod(0o600)
                except Exception:
                    pass
        # Köhnə env var formatı üçün də set edirik
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = api_token
        return True

    # Lokal inkişaf — kaggle.json artıq varsa
    for yol in [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ]:
        if yol.exists():
            return True

    return False


def _kaggle_api():
    """
    Konfiq hazırlanandan sonra Kaggle SDK-nı import edib authenticate edilmiş API qaytarır.
    LAZY import — modul-leveldə deyil, yalnız çağırılanda import edir.
    """
    if not _kaggle_hazirla():
        raise RuntimeError(
            "Kaggle konfiqurasyonu tapılmadı!\n"
            "Railway Variables-a əlavə edin:\n"
            "  KAGGLE_API_TOKEN = KGAT_...\n"
            "  KAGGLE_USERNAME  = kaggle_istifadeci_adiniz"
        )

    # Lazy import — yalnız bu nöqtədə import edirik
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _config_yukle() -> dict:
    """Mövcud Kaggle konfiqini qaytarır."""
    api_token = os.environ.get("KAGGLE_API_TOKEN", "")
    username = os.environ.get("KAGGLE_USERNAME", "")
    if api_token and username:
        return {"username": username, "key": api_token}

    for yol in [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ]:
        if yol.exists():
            with open(yol) as f:
                return json.load(f)
    return {}


def _notebook_slug() -> str:
    """
    Kaggle-dakı notebook-un slug-ını qaytarır.
    Railway-də: KAGGLE_NOTEBOOK_SLUG env var-ından oxuyur.
    Kaggle URL-dən götürün: kaggle.com/code/USERNAME/NOTEBOOK-NAME/edit
    """
    cfg = _config_yukle()
    username = cfg.get("username", os.environ.get("KAGGLE_USERNAME", ""))
    notebook_adi = os.environ.get("KAGGLE_NOTEBOOK_SLUG", "notebookbc36ec01da")
    return f"{username}/{notebook_adi}"


# ─────────────────────────────────────────────────────────────
# ANA FUNKSIYALAR
# ─────────────────────────────────────────────────────────────

def is_gondər(video_bytes: bytes, prompt: str, fayl_adi: str = "video.mp4") -> str:
    """
    TAM AVTOMATİK: Video upload → dataset yarat → kernel işlət
    Qaytarır: iş ID-si (polling üçün)
    """
    is_id = str(uuid.uuid4())[:8]

    is_qovluqu = Path("video_jobs") / is_id
    is_qovluqu.mkdir(parents=True, exist_ok=True)

    video_yolu = is_qovluqu / fayl_adi
    with open(video_yolu, "wb") as f:
        f.write(video_bytes)

    with open(is_qovluqu / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    _is_leyi[is_id] = {
        "status": "uploading",
        "prompt": prompt,
        "video_path": str(video_yolu),
        "output_path": None,
        "dataset_slug": None,
        "created_at": time.time()
    }

    print(f"[Kaggle] Yeni iş: {is_id} | Prompt: {prompt[:50]}...")

    import threading
    t = threading.Thread(
        target=_arxa_planda_isle,
        args=(is_id, is_qovluqu, prompt, fayl_adi),
        daemon=True
    )
    t.start()
    return is_id


def _arxa_planda_isle(is_id: str, is_qovluqu: Path, prompt: str, fayl_adi: str):
    """Arxa planda: video upload → kernel işlət → nəticə yüklə"""
    try:
        api = _kaggle_api()  # Lazy import burada baş verir
        cfg = _config_yukle()
        username = cfg.get("username", "")

        # ── 1. Dataset yarat & upload ─────────────────────────
        _is_leyi[is_id]["status"] = "uploading"
        print(f"[Kaggle {is_id}] Video Kaggle-a yüklənir...")

        dataset_adi = f"ismayilai-video-{is_id}"
        dataset_slug = f"{username}/{dataset_adi}"
        _is_leyi[is_id]["dataset_slug"] = dataset_slug

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(is_qovluqu / fayl_adi, tmp_path / fayl_adi)
            with open(tmp_path / "prompt.txt", "w") as f:
                f.write(prompt)
            meta = {
                "title": f"İsmayıl AI Video {is_id}",
                "id": dataset_slug,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(tmp_path / "dataset-metadata.json", "w") as f:
                json.dump(meta, f)
            api.dataset_create_new(
                folder=str(tmp_path),
                dir_mode="zip",
                convert_to_csv=False,
                public=False
            )

        print(f"[Kaggle {is_id}] Dataset yükləndi: {dataset_slug}")
        time.sleep(15)  # Dataset process olunana qədər gözlə

        # ── 2. Kernel-i işlət ────────────────────────────────
        _is_leyi[is_id]["status"] = "running"
        kernel_slug = _notebook_slug()

        kernel_meta = {
            "id": kernel_slug,
            "title": "İsmayıl AI Video Worker",
            "code_file": "video_edit_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [dataset_slug],
            "competition_sources": [],
            "kernel_sources": []
        }

        with tempfile.TemporaryDirectory() as ktmp:
            ktmp_path = Path(ktmp)
            worker_kod = _worker_kodu_yarat(dataset_slug, fayl_adi, prompt)
            with open(ktmp_path / "video_edit_worker.py", "w", encoding="utf-8") as f:
                f.write(worker_kod)
            with open(ktmp_path / "kernel-metadata.json", "w") as f:
                json.dump(kernel_meta, f, indent=2)
            api.kernels_push(str(ktmp_path))

        print(f"[Kaggle {is_id}] Kernel push edildi, işlənir...")

        # ── 3. Kernel tamamlanana qədər gözlə ───────────────
        _kernel_bitene_qeder_gozle(api, kernel_slug, is_id)

        # ── 4. Nəticəni yüklə ───────────────────────────────
        if _is_leyi[is_id]["status"] == "done":
            _netice_yukle(api, kernel_slug, is_id)

    except Exception as xata:
        print(f"[Kaggle {is_id}] XƏTA: {xata}")
        _is_leyi[is_id]["status"] = "error"
        _is_leyi[is_id]["error"] = str(xata)


def _kernel_bitene_qeder_gozle(api, kernel_slug: str, is_id: str, max_deqiqe: int = 60):
    """Kernel tamamlanana qədər hər 30 saniyədən bir yoxlayır."""
    baslangic = time.time()
    max_saniye = max_deqiqe * 60

    while True:
        try:
            username, kernel_adi = kernel_slug.split("/")
            cavab = api.process_response(
                api.kernels_status_with_http_info(username, kernel_adi)
            )
            status = cavab.get("status", "")
            print(f"[Kaggle {is_id}] Kernel status: {status}")

            if status == "complete":
                _is_leyi[is_id]["status"] = "done"
                return
            elif status in ("error", "cancelAcknowledged"):
                _is_leyi[is_id]["status"] = "error"
                _is_leyi[is_id]["error"] = f"Kaggle kernel xətası: {status}"
                return
        except Exception as e:
            print(f"[Kaggle {is_id}] Status sorğusu xətası: {e}")

        if time.time() - baslangic > max_saniye:
            _is_leyi[is_id]["status"] = "error"
            _is_leyi[is_id]["error"] = "Vaxt limiti aşıldı (60 dəq)"
            return
        time.sleep(30)


def _netice_yukle(api, kernel_slug: str, is_id: str):
    """Tamamlanmış kernel-in output fayllarını yükləyir."""
    try:
        cixis_qovluqu = Path("video_jobs") / is_id
        username, kernel_adi = kernel_slug.split("/")
        api.kernels_output(username, kernel_adi, path=str(cixis_qovluqu))

        for zip_fayl in cixis_qovluqu.glob("*.zip"):
            with zipfile.ZipFile(zip_fayl, "r") as z:
                z.extractall(cixis_qovluqu)
            zip_fayl.unlink()

        for mp4 in cixis_qovluqu.glob("output*.mp4"):
            hedef = cixis_qovluqu / "output.mp4"
            mp4.rename(hedef)
            _is_leyi[is_id]["output_path"] = str(hedef)
            print(f"[Kaggle {is_id}] ✅ Video hazır: {hedef}")
            break
    except Exception as e:
        print(f"[Kaggle {is_id}] Nəticə yükləmə xətası: {e}")
        _is_leyi[is_id]["status"] = "error"
        _is_leyi[is_id]["error"] = str(e)


def _worker_kodu_yarat(dataset_slug: str, fayl_adi: str, prompt: str) -> str:
    """video_edit_worker.py-dan oxuyub, INPUT_VIDEO_YOLU və PROMPT-u dinamik dəyişir."""
    dataset_adi = dataset_slug.split("/")[-1]
    worker = Path(__file__).parent.parent / "kaggle_notebook" / "video_edit_worker.py"
    if not worker.exists():
        worker = Path(__file__).parent / "kaggle_notebook" / "video_edit_worker.py"
    kod = worker.read_text(encoding="utf-8")
    kod = kod.replace(
        'INPUT_VIDEO_YOLU = "/kaggle/input/my-video/video.mp4"',
        f'INPUT_VIDEO_YOLU = "/kaggle/input/{dataset_adi}/{fayl_adi}"'
    )
    kod = kod.replace(
        'PROMPT = "oil painting style, impressionist, colorful brushstrokes, masterpiece"',
        f'PROMPT = "{prompt}"'
    )
    kod = kod.replace(
        'CIXIS_YOLU = "/kaggle/working/output_video.mp4"',
        'CIXIS_YOLU = "/kaggle/working/output.mp4"'
    )
    return kod


# ─────────────────────────────────────────────────────────────
# STATUS VƏ SİYAHI
# ─────────────────────────────────────────────────────────────

def is_veziyyeti(is_id: str) -> dict:
    """İş vəziyyətini qaytarır."""
    if is_id not in _is_leyi:
        return {"status": "not_found", "message": "İş tapılmadı"}

    melu = _is_leyi[is_id]
    cixis = Path("video_jobs") / is_id / "output.mp4"
    if cixis.exists() and melu["status"] not in ("done", "error"):
        _is_leyi[is_id]["status"] = "done"
        _is_leyi[is_id]["output_path"] = str(cixis)

    cavab = {
        "status": melu["status"],
        "prompt": melu["prompt"],
        "job_id": is_id,
        "elapsed_sec": round(time.time() - melu["created_at"])
    }
    if melu.get("error"):
        cavab["error"] = melu["error"]
    if melu["status"] == "done":
        cavab["video_url"] = f"/video/download/{is_id}"
    return cavab


def is_siyahisi() -> list:
    """Bütün işlərin siyahısını qaytarır."""
    return [{"job_id": k, **v} for k, v in _is_leyi.items()]
