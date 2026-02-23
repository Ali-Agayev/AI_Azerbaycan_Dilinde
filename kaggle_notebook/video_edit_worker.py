"""
╔══════════════════════════════════════════════════════════════════╗
║     İsmayıl AI — Kaggle Video Düzəltmə Worker                  ║
║     Stable Diffusion img2img | Frame-by-Frame Emal              ║
║                                                                  ║
║  NECƏ İSTİFADƏ ETMƏLİ:                                         ║
║  1. Bu faylı Kaggle Notebook-a kopyalayın                       ║
║  2. Settings → Accelerator = GPU T4                             ║
║  3. Settings → Internet = ON                                    ║
║  4. INPUT_VIDEO_YOLU və PROMPT dəyişənlərini doldurun           ║
║  5. Bütün hüceyrələri ardıcıl işlədin                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# HÜCEYRƏ 1 — GPU YOXLAMASI
# ═══════════════════════════════════════════════════════════════════
import torch

print("=" * 50)
print("CUDA mövcuddurmu?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Yaddaşı:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print("⚠️  GPU tapılmadı! Settings → Accelerator = GPU T4 seçin!")
print("=" * 50)


# ═══════════════════════════════════════════════════════════════════
# HÜCEYRƏ 2 — PAKETLƏRİN QURAŞDIRILMASI
# ═══════════════════════════════════════════════════════════════════
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "diffusers==0.27.2",
    "transformers==4.40.0",
    "accelerate==0.30.1",
    "xformers",
    "opencv-python-headless",
    "imageio[ffmpeg]",
    "Pillow"
], check=True)
print("✅ Bütün paketlər quraşdırıldı!")


# ═══════════════════════════════════════════════════════════════════
# HÜCEYRƏ 3 — MODELİN YÜKLƏNMƏSİ (HuggingFace-dən)
# ═══════════════════════════════════════════════════════════════════
from diffusers import StableDiffusionImg2ImgPipeline
import torch

print("🔄 Model yüklənir... (ilk dəfə ~3-5 dəqiqə çəkə bilər)")

boru_kəməri = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",   # Pulsuz, açıq mənbəli model
    torch_dtype=torch.float16,           # Yarım precision — daha sürətli
    safety_checker=None,                 # NSFW qorumasını söndürürük (sürət üçün)
    requires_safety_checker=False
)

# GPU-ya köçürürük
boru_kəməri = boru_kəməri.to("cuda")

# Yaddaş optimallaşdırması — T4 üçün vacibdir
boru_kəməri.enable_xformers_memory_efficient_attention()

print("✅ Stable Diffusion v1.5 uğurla yükləndi!")
print(f"   GPU Yaddaşı İstifadəsi: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════
# HÜCEYRƏ 4 — KONFIQURASIYA (BURAYA BAXIN!)
# ═══════════════════════════════════════════════════════════════════

# ⬇️  BU İKİ DƏYİŞƏNİ ÖZÜNÜZ DOLDURUN:
INPUT_VIDEO_YOLU = "/kaggle/input/my-video/video.mp4"   # Kaggle Dataset-dən yükləyin
PROMPT = "oil painting style, impressionist, colorful brushstrokes, masterpiece"

# Əlavə tənzimləmələr:
GÜCLÜLÜK = 0.6        # 0.0 (dəyişiklik yoxdur) → 1.0 (tam fərqli şəkil). 0.5-0.7 tövsiyə edilir
ADIM_SAYI = 25        # Diffusion addımları. Az = sürətli, çox = keyfiyyətli (15-30)
REHBERLIK = 7.5       # Prompt-a nə qədər riayət etsin (6-9 arası yaxşıdır)
FPS_LIMIT = 8         # Saniyədə frame sayı (çox fps = çox GPU vaxtı)
MAX_EN = 512          # Şəkil eni (SD v1.5 üçün 512 ideal)
MAX_BOY = 512         # Şəkil boyu (SD v1.5 üçün 512 ideal)
CIXIS_YOLU = "/kaggle/working/output_video.mp4"

print(f"📋 Konfiqurasiya:")
print(f"   Giriş video: {INPUT_VIDEO_YOLU}")
print(f"   Prompt: {PROMPT}")
print(f"   Güclülük: {GÜCLÜLÜK}")
print(f"   FPS limiti: {FPS_LIMIT}")


# ═══════════════════════════════════════════════════════════════════
# HÜCEYRƏ 5 — VİDEO EMAL FUNKSİYALARI
# ═══════════════════════════════════════════════════════════════════
import cv2
import numpy as np
from PIL import Image
import imageio
from pathlib import Path

def video_frame_lere_bol(video_yolu: str, fps_limiti: int = 8):
    """
    Videonu frame-lərə bölür.
    fps_limiti: saniyədə nə qədər frame emal edilsin (GPU vaxtına qənaət üçün)
    """
    cap = cv2.VideoCapture(video_yolu)
    
    if not cap.isOpened():
        raise ValueError(f"Video açıla bilmədi: {video_yolu}")
    
    orijinal_fps = cap.get(cv2.CAP_PROP_FPS)
    umumi_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Hər neçə frame-dən birini götürəcəyimizi hesablayırıq
    atlama = max(1, int(orijinal_fps / fps_limiti))
    
    print(f"📹 Video məlumatları:")
    print(f"   Orijinal FPS: {orijinal_fps:.1f}")
    print(f"   Cəmi frame: {umumi_frame}")
    print(f"   Emal ediləcək frame: ~{umumi_frame // atlama}")
    print(f"   Frame atlama: hər {atlama}-cidən biri")
    
    frame_ler = []
    frame_sayac = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_sayac % atlama == 0:
            # BGR → RGB (OpenCV BGR, PIL isə RGB istifadə edir)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            
            # Ölçüləndiririk (SD üçün)
            pil_frame = pil_frame.resize(
                (MAX_EN, MAX_BOY), 
                Image.LANCZOS
            )
            frame_ler.append(pil_frame)
        
        frame_sayac += 1
    
    cap.release()
    print(f"✅ {len(frame_ler)} frame oxundu")
    return frame_ler, fps_limiti


def frame_leri_emalet(frame_ler: list, prompt: str) -> list:
    """
    Hər frame-ə Stable Diffusion img2img tətbiq edir.
    Bu, əsas video düzəltmə addımıdır.
    """
    emallenmis_frame_ler = []
    
    print(f"\n🎨 Video emal başlayır...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Emal ediləcək frame sayı: {len(frame_ler)}")
    print("-" * 40)
    
    for i, frame in enumerate(frame_ler):
        print(f"   Frame {i+1}/{len(frame_ler)} emal edilir...", end="\r")
        
        with torch.autocast("cuda"):
            nəticə = boru_kəməri(
                prompt=prompt,
                image=frame,
                strength=GÜCLÜLÜK,
                num_inference_steps=ADIM_SAYI,
                guidance_scale=REHBERLIK,
            )
        
        emallenmis_frame_ler.append(nəticə.images[0])
        
        # Hər 10 frame-dən sonra GPU yaddaşını təmizləyirik
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n✅ {len(emallenmis_frame_ler)} frame uğurla emal edildi!")
    return emallenmis_frame_ler


def frame_leri_videoya_birlesdır(frame_ler: list, cixis_yolu: str, fps: int = 8):
    """
    Emal olunmuş frame-ləri video faylına birləşdirir.
    """
    print(f"\n🎬 Frameler vidoya birləşdirilir...")
    
    # PIL Image-ləri numpy array-ə çeviririk
    numpy_frame_ler = [np.array(f) for f in frame_ler]
    
    # imageio ilə video yazırıq
    writer = imageio.get_writer(
        cixis_yolu, 
        fps=fps,
        codec='libx264',
        quality=8
    )
    
    for frame in numpy_frame_ler:
        writer.append_data(frame)
    
    writer.close()
    
    # Nəticə faylının ölçüsünü yoxlayırıq
    cixis = Path(cixis_yolu)
    olchu = cixis.stat().st_size / 1e6  # MB
    
    print(f"✅ Video yaradıldı: {cixis_yolu}")
    print(f"   Fayl ölçüsü: {olchu:.1f} MB")
    return cixis_yolu


# ═══════════════════════════════════════════════════════════════════
# HÜCEYRƏ 6 — ANA EMAL (BURAYA BAXIN!)
# ═══════════════════════════════════════════════════════════════════
import time

baslangic = time.time()

print("🚀 Video emal prosesi başlayır!")
print("=" * 50)

# Addım 1: Videonu frame-lərə böl
frame_ler, cixis_fps = video_frame_lere_bol(INPUT_VIDEO_YOLU, FPS_LIMIT)

# Addım 2: Hər frame-i Stable Diffusion ilə emal et
emallenmis = frame_leri_emalet(frame_ler, PROMPT)

# Addım 3: Frame-ləri yenidən videoya birləşdir
frame_leri_videoya_birlesdır(emallenmis, CIXIS_YOLU, cixis_fps)

bitme = time.time()
kecen = round(bitme - baslangic, 1)

print("\n" + "=" * 50)
print(f"🎉 TAMAMLANDI! Cəmi vaxt: {kecen} saniyə ({kecen/60:.1f} dəqiqə)")
print(f"📁 Nəticə video: {CIXIS_YOLU}")
print("=" * 50)
print("\n💡 Nəticəni yükləmək üçün:")
print("   Kaggle → sağ panel → 'Output' → 'output_video.mp4' → Download")


# ═══════════════════════════════════════════════════════════════════
# HÜCEYRƏ 7 — NƏTİCƏNİ GÖSTƏR (Notebook-da preview)
# ═══════════════════════════════════════════════════════════════════
from IPython.display import Video, display

print("📺 Nəticə videonun preview-i:")
display(Video(CIXIS_YOLU, embed=True, width=640))
