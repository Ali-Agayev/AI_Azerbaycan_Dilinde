"""
kaggle_client.py — TAM AVTOMATİK Kaggle API İnteqrasiyası
==========================================================
İş axını:
  1. Video faylı Kaggle Dataset kimi upload edir
  2. Notebook-u (kernel) Kaggle API vasitəsilə işlədır
  3. Kernel tamamlanana qədər status izləyir
  4. Nəticə videonu avtomatik yükləyir

Tələb: ~/.kaggle/kaggle.json  ←  Kaggle API açarı
"""

import os
import json
import time
import uuid
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# kaggle SDK-sı
try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    KAGGLE_SDK_VAR = True
except ImportError:
    KAGGLE_SDK_VAR = False

# İş vəziyyəti (yaddaşda saxlanılır)
_is_leyi: dict = {}

# ─────────────────────────────────────────────────────────────
# KONFIQURASIYA — Bunları Kaggle hesabınıza uyğun dəyişin!
# ─────────────────────────────────────────────────────────────
def _config_yukle() -> dict:
    """
    Kaggle API məlumatlarını oxər. İki format dəstəklənir:

    YENİ format (KGAT_ prefiksi):
        KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxx

    KÖHNƏ format (legacy):
        KAGGLE_USERNAME=... + KAGGLE_KEY=...
        və ya ~/.kaggle/kaggle.json
    """
    # ─ YENİ: tək token (KGAT_...) ──────────────────────────
    api_token = os.environ.get("KAGGLE_API_TOKEN", "")
    if api_token:
        os.environ["KAGGLE_API_TOKEN"] = api_token  # SDK üçün
        # Username-i ayri env var-dan al (və ya URL-dən)
        username = os.environ.get("KAGGLE_USERNAME", "")
        return {"username": username, "key": api_token, "token": api_token}

    # ─ KÖHNƏ: username + key ─────────────────────────────
    env_username = os.environ.get("KAGGLE_USERNAME", "")
    env_key = os.environ.get("KAGGLE_KEY", "")
    if env_username and env_key:
        acar_qovluqu = Path.home() / ".kaggle"
        acar_qovluqu.mkdir(exist_ok=True)
        acar_yolu = acar_qovluqu / "kaggle.json"
        if not acar_yolu.exists():
            with open(acar_yolu, "w") as f:
                json.dump({"username": env_username, "key": env_key}, f)
            acar_yolu.chmod(0o600)
        return {"username": env_username, "key": env_key}

    # ─ LOKAL: ~/.kaggle/kaggle.json ──────────────────────────
    acar_yolu = Path.home() / ".kaggle" / "kaggle.json"
    if acar_yolu.exists():
        with open(acar_yolu) as f:
            return json.load(f)

    return {}


def _kaggle_api() -> "KaggleApiExtended":
    """Authenticate edilmiş Kaggle API obyekti qaytarır."""
    # Yeni token formatı env var-da varırsa, SDK-ya bildiririk
    if os.environ.get("KAGGLE_API_TOKEN"):
        os.environ.setdefault("KAGGLE_API_TOKEN", os.environ["KAGGLE_API_TOKEN"])
    api = KaggleApiExtended()
    api.authenticate()
    return api

# ─────────────────────────────────────────────────────────────
# NOTEBOOK SLUG — Kaggle-da yaratdığınız notebook-un URL-i
# ─────────────────────────────────────────────────────────────
def _notebook_slug() -> str:
    """
    Kaggle-dakı notebook-un slug-ını qaytarır.
    
    Railway-də: KAGGLE_NOTEBOOK_SLUG env var-ından oxuyur
    Lokal: ~/.kaggle/kaggle.json + KAGGLE_NOTEBOOK_SLUG
    
    Kaggle URL-dən götürün:
    kaggle.com/code/USERNAME/NOTEBOOK-NAME/edit
                             ↑ NOTEBOOK-NAME hissəsi
    """
    cfg = _config_yukle()
    username = cfg.get("username", os.environ.get("KAGGLE_USERNAME", ""))
    # Env var-dan notebook adını oxuyuruq
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
    if not KAGGLE_SDK_VAR:
        raise RuntimeError(
            "Kaggle SDK quraşdırılmayıb. Backend terminalında işlədin:\n"
            "  pip install kaggle\n"
            "Sonra serveri yenidən başladın."
        )

    is_id = str(uuid.uuid4())[:8]

    # ── İş qovluğunu hazırla ──────────────────────────────────
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

    # ── Arxa planda işi başlat (async deyil — thread ilə) ────
    import threading
    t = threading.Thread(target=_arxa_planda_isle, args=(is_id, is_qovluqu, prompt, fayl_adi), daemon=True)
    t.start()

    return is_id


def _arxa_planda_isle(is_id: str, is_qovluqu: Path, prompt: str, fayl_adi: str):
    """
    Arxa planda:
      1. Video faylı Kaggle Dataset-ə yükləyir
      2. Kernel-i (notebook) işlədır
      3. Nəticəni yükləyir
    """
    try:
        api = _kaggle_api()
        cfg = _config_yukle()
        username = cfg.get("username", "")

        # ── 1. Dataset yarat & upload ────────────────────────
        _is_leyi[is_id]["status"] = "uploading"
        print(f"[Kaggle {is_id}] Video Kaggle-a yüklənir...")

        dataset_adi = f"ismayilai-video-{is_id}"
        dataset_slug = f"{username}/{dataset_adi}"
        _is_leyi[is_id]["dataset_slug"] = dataset_slug

        # Müvəqqəti qovluqda dataset metadata yazırıq
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Video faylını kopyalayırıq
            shutil.copy(is_qovluqu / fayl_adi, tmp_path / fayl_adi)

            # Prompt faylını da əlavə edirik
            with open(tmp_path / "prompt.txt", "w") as f:
                f.write(prompt)

            # Dataset metadata (dataset-metadata.json)
            meta = {
                "title": f"İsmayıl AI Video {is_id}",
                "id": dataset_slug,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(tmp_path / "dataset-metadata.json", "w") as f:
                json.dump(meta, f)

            # Dataset-i Kaggle-a yükləyirik (yeni yaradırıq)
            api.dataset_create_new(
                folder=str(tmp_path),
                dir_mode="zip",
                convert_to_csv=False,
                public=False
            )

        print(f"[Kaggle {is_id}] Dataset yükləndi: {dataset_slug}")

        # Kernel hazır olana qədər bir az gözlə (dataset process olunur)
        time.sleep(15)

        # ── 2. Kernel-i işlət ───────────────────────────────
        _is_leyi[is_id]["status"] = "running"
        print(f"[Kaggle {is_id}] Notebook işlədilir...")

        kernel_slug = _notebook_slug()

        # Kernel konfiqurasiyasını yeniləyirik (yeni dataset + prompt)
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

        # Kernel-i API vasitəsilə yenidən işlədırik
        with tempfile.TemporaryDirectory() as ktmp:
            ktmp_path = Path(ktmp)

            # Dinamik prompt ilə worker skriptini yenilə
            worker_kod = _worker_kodu_yarat(
                dataset_slug=dataset_slug,
                fayl_adi=fayl_adi,
                prompt=prompt
            )
            with open(ktmp_path / "video_edit_worker.py", "w", encoding="utf-8") as f:
                f.write(worker_kod)

            with open(ktmp_path / "kernel-metadata.json", "w") as f:
                json.dump(kernel_meta, f, indent=2)

            # Kernel push et (bu həm yaradır, həm işlədır)
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


def _kernel_bitene_qeder_gozle(api, kernel_slug: str, is_id: str, max_dəqiqə: int = 60):
    """
    Kernel-in tamamlanmasını gözləyir (hər 30 saniyədən bir yoxlayır).
    Maksimum gözləmə: 60 dəqiqə (Kaggle session limiti).
    """
    baslangic = time.time()
    max_saniye = max_dəqiqə * 60

    while True:
        try:
            # Kernel statusunu yoxlayırıq
            username, kernel_adi = kernel_slug.split("/")
            cavab = api.process_response(
                api.kernels_status_with_http_info(username, kernel_adi)
            )
            status = cavab.get("status", "")
            print(f"[Kaggle {is_id}] Kernel status: {status}")

            if status == "complete":
                _is_leyi[is_id]["status"] = "done"
                print(f"[Kaggle {is_id}] ✅ Kernel tamamlandı!")
                return
            elif status in ("error", "cancelAcknowledged"):
                _is_leyi[is_id]["status"] = "error"
                _is_leyi[is_id]["error"] = f"Kaggle kernel xətası: {status}"
                print(f"[Kaggle {is_id}] ❌ Kernel xəta: {status}")
                return

        except Exception as e:
            print(f"[Kaggle {is_id}] Status sorğusu xətası: {e}")

        # Vaxt limiti yoxlanışı
        if time.time() - baslangic > max_saniye:
            _is_leyi[is_id]["status"] = "error"
            _is_leyi[is_id]["error"] = "Vaxt limiti aşıldı (60 dəq)"
            return

        time.sleep(30)  # Hər 30 saniyədən bir yoxla


def _netice_yukle(api, kernel_slug: str, is_id: str):
    """Tamamlanmış kernel-in output fayllarını yükləyir."""
    try:
        cixis_qovluqu = Path("video_jobs") / is_id
        username, kernel_adi = kernel_slug.split("/")

        print(f"[Kaggle {is_id}] Nəticə yüklənir...")

        # Kernel output-larını yüklə (ZIP kimi gəlir)
        api.kernels_output(username, kernel_adi, path=str(cixis_qovluqu))

        # ZIP-i açırıq (əgər varsa)
        for zip_fayl in cixis_qovluqu.glob("*.zip"):
            with zipfile.ZipFile(zip_fayl, "r") as z:
                z.extractall(cixis_qovluqu)
            zip_fayl.unlink()

        # output_video.mp4 faylını output.mp4 adlandırırıq
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
    """
    video_edit_worker.py faylından oxuyub,
    INPUT_VIDEO_YOLU və PROMPT-u dinamik şəkildə dəyişir.
    """
    dataset_adi = dataset_slug.split("/")[-1]
    worker = Path(__file__).parent.parent / "kaggle_notebook" / "video_edit_worker.py"

    if worker.exists():
        kod = worker.read_text(encoding="utf-8")
    else:
        # Fallback — əsas kod
        kod = open(Path(__file__).parent / "kaggle_notebook" / "video_edit_worker.py").read()

    # Dinamik olaraq dataset yolunu və prompt-u dəyişirik
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

    # Çıxış faylını yoxlayırıq
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
