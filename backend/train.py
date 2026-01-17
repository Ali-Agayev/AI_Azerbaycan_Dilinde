import torch
import torch.nn as nn
from model import IsmayilModeli, blok_olcusu # Model memarlığını və blok ölçüsünü gətiririk
from tokenizer import CharTokenizator      # Tokenizator sinfini gətiririk
import os

# Təlim üçün Hiperparametrlər
paket_olcusu = 32         # batch_size: Hər addımda neçə cümlə eyni vaxtda öyrəniləcək
maks_iterasiya = 1000     # max_iters: Toplam neçə dəfə öyrənmə addımı atılacaq
qiymetlendirme_araligi = 500 # eval_interval: Hər neçə addımdan bir modelin vəziyyəti yoxlanılacaq
oyrenme_derecesi = 1e-3   # learning_rate: Modelin səhvlərindən nə qədər sürətlə nəticə çıxaracağı
qiymetlendirme_sayi = 200 # eval_iters: Yoxlama zamanı ortalama hesablamaq üçün istifadə olunan addım sayı
cihaz = 'cuda' if torch.cuda.is_available() else 'cpu' # Əgər NVIDIA kartı varsa CUDA, yoxdursa CPU istifadə olunur

# Verilənləri (məlumat bazasını) yükləyirik
path = 'input.txt'
if not os.path.exists(path):
    print(f"Səhv: {path} faylı tapılmadı!")
    exit()

with open(path, 'r', encoding='utf-8') as f:
    metn = f.read()

# Tokenizatoru yaradırıq və lüğəti yadda saxlayırıq
tokenizator = CharTokenizator(metn)
tokenizator.yadda_saxla('tokenizer.json')
luget_olcusu = tokenizator.luget_olcusu

# Məlumatın hamısını rəqəmlərə çevirib gərginlik (tensor) halına salırıq
məlumat = torch.tensor(tokenizator.kodlasdir(metn), dtype=torch.long)

# Məlumatı Tədris (90%) və Yoxlama (10%) hissələrinə ayırırıq
n = int(0.9 * len(məlumat))
tedris_melumati = məlumat[:n]
yoxlama_melumati = məlumat[n:]

# Model üçün təsadüfi paketlər (batches) hazırlayan funksiya
def paket_getir(bolme):
    # Hansı hissədən məlumat götürəcəyimizi seçirik
    cari_melumat = tedris_melumati if bolme == 'train' else yoxlama_melumati
    # Təsadüfi başlanğıc nöqtələri seçirik
    bashlangiclar = torch.randint(len(cari_melumat) - blok_olcusu, (paket_olcusu,))
    # Giriş (x) və hədəf (y) məlumatlarını hazırlayırıq (y hər zaman x-dən bir addım öndədir)
    x = torch.stack([cari_melumat[i:i+blok_olcusu] for i in bashlangiclar])
    y = torch.stack([cari_melumat[i+1:i+blok_olcusu+1] for i in bashlangiclar])
    x, y = x.to(cihaz), y.to(cihaz)
    return x, y

# Modelin itkisini (səhvini) təxmin edən funksiya
@torch.no_grad()
def itkini_təxmin_et(model):
    sonuc = {}
    model.eval() # Modeli qiymətləndirmə rejiminə keçiririk
    for bolme in ['train', 'val']:
        itkiler = torch.zeros(qiymetlendirme_sayi)
        for k in range(qiymetlendirme_sayi):
            X, Y = paket_getir(bolme)
            ehtimallar, itki = model(X, Y)
            itkiler[k] = itki.item()
        sonuc[bolme] = itkiler.mean() # Ortalama itkini qeyd edirik
    model.train() # Modeli yenidən təlim rejiminə qaytarırıq
    return sonuc

# Modeli başladırıq
ismayil = IsmayilModeli(luget_olcusu)
ismayil = ismayil.to(cihaz)

# Optimallaşdırıcı (AdamW - Adam with Weight Decay)
# Bu alət modelin çəkilərini səhvlərə uyğun olaraq tənzimləyir
optimallashdirici = torch.optim.AdamW(ismayil.parameters(), lr=oyrenme_derecesi)

print(f"İsmayıl {cihaz} üzərində öyrənməyə başlayır...")

for addim in range(maks_iterasiya):
    # Müəyyən aralıqlarla ekrana hesabat veririk
    if addim % qiymetlendirme_araligi == 0:
        itkiler = itkini_təxmin_et(ismayil)
        print(f"Addım {addim}: Tədris itkisi {itkiler['train']:.4f}, Yoxlama itkisi {itkiler['val']:.4f}")

    # Öyrənmək üçün bir paket məlumat götürürük
    xb, yb = paket_getir('train')

    # İrəli ötürmə və itki hesablama
    ehtimallar, itki = ismayil(xb, yb)
    optimallashdirici.zero_grad(set_to_none=True) # Köhnə qradiyentləri silirik
    itki.backward() # Geri ötürmə (Backpropagation) - Səhvi hesabla
    optimallashdirici.step() # Çəkiləri yenilə (Update weights)

# Təlim bitdikdən sonra modelin "beynini" (çəkilərini) yadda saxlayırıq
torch.save(ismayil.state_dict(), 'ismayil_model.pth')
print("Təlim tamamlandı! Model 'ismayil_model.pth' faylına yazıldı.")

# Test üçün bir neçə söz yaradaq
bashlangic = torch.zeros((1, 1), dtype=torch.long, device=cihaz)
print("\nİsmayılın ilk cümlələri:")
print(tokenizator.de_kodlasdir(ismayil.yeni_metn_yarat(bashlangic, maksimum_yeni_simvol=100)[0].tolist()))
