import torch
import torch.nn as nn
from torch.nn import functional as F

# Hiperparametrlər (Modelin ölçüsünü və gücünü müəyyən edən sabitlər)
yerlesdirme_olcusu = 128 # n_embd: Hər bir simbolun neçə rəqəmlə təmsil olunacağı
bash_sayi = 4            # n_head: Diqqət mexanizminin neçə paralel hissədən ibarət olacağı
lay_sayi = 4             # n_layer: Transformer bloklarının sayı (dərinlik)
atilan_melumat = 0.1     # dropout: Modelin çox öyrənib əzbərləməməsi üçün bəzi məlumatları təsadüfi silmə dərəcəsi
blok_olcusu = 64         # block_size: Bir dəfəyə baxılan maksimum mətni uzunluğu (kontekst)

class DiqqetBashi(nn.Module):
    """ 
    Bu sinif "Self-Attention" (Özünə Diqqət) mexanizminin bir hissəsidir.
    AI mətn içindəki sözlərin bir-biri ilə necə əlaqəli olduğunu bura vasitəsilə başa düşür.
    """
    def __init__(self, bash_olcusu):
        super().__init__()
        self.acharlar = nn.Linear(yerlesdirme_olcusu, bash_olcusu, bias=False) # Key
        self.sorgular = nn.Linear(yerlesdirme_olcusu, bash_olcusu, bias=False) # Query
        self.deyerler = nn.Linear(yerlesdirme_olcusu, bash_olcusu, bias=False) # Value
        # Gələcəkdəki sözləri görməmək üçün maskalama üçbucağı (Lower triangular matrix)
        self.register_buffer('maska', torch.tril(torch.ones(blok_olcusu, blok_olcusu)))
        self.seyriltme = nn.Dropout(atilan_melumat)

    def forward(self, x):
        B, T, C = x.shape
        k = self.acharlar(x)   # (B, T, bash_olcusu)
        q = self.sorgular(x) # (B, T, bash_olcusu)
        
        # Diqqət ballarının hesablanması (Hansı söz hansına daha çox fokuslanmalıdır?)
        bali = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        bali = bali.masked_fill(self.maska[:T, :T] == 0, float('-inf')) # Gələcəyi bağla
        bali = F.softmax(bali, dim=-1) # Ehtimallara çevir
        bali = self.seyriltme(bali)
        
        # Dəyərlərin bu ballar əsasında birləşdirilməsi
        v = self.deyerler(x) # (B, T, bash_olcusu)
        sonuc = bali @ v    # (B, T, T) @ (B, T, bash_olcusu) -> (B, T, bash_olcusu)
        return sonuc

class ChoxBashliDiqqet(nn.Module):
    """ Bir neçə diqqət başlığının paralel işləməsini təmin edir. """
    def __init__(self, bash_sayi, bash_olcusu):
        super().__init__()
        self.bashlar = nn.ModuleList([DiqqetBashi(bash_olcusu) for _ in range(bash_sayi)])
        self.proyeksiya = nn.Linear(yerlesdirme_olcusu, yerlesdirme_olcusu)
        self.seyriltme = nn.Dropout(atilan_melumat)

    def forward(self, x):
        # Bütün başlıqların nəticələrini yan-yana düzürük
        sonuc = torch.cat([b(x) for b in self.bashlar], dim=-1)
        sonuc = self.seyriltme(self.proyeksiya(sonuc))
        return sonuc

class IreliBesleme(nn.Module):
    """ Modelin öyrəndiyi məlumatları emal etməsi üçün sadə neyron şəbəkə qatı. """
    def __init__(self, yerlesdirme_olcusu):
        super().__init__()
        self.shabaka = nn.Sequential(
            nn.Linear(yerlesdirme_olcusu, 4 * yerlesdirme_olcusu),
            nn.ReLU(), # Aktivləşdirmə funksiyası (Qeyri-xəttilik əlavə edir)
            nn.Linear(4 * yerlesdirme_olcusu, yerlesdirme_olcusu),
            nn.Dropout(atilan_melumat),
        )

    def forward(self, x):
        return self.shabaka(x)

class TransformerBloku(nn.Module):
    """ 
    Əsas Transformer kərpici: 
    Əvvəlcə ünsiyyət (Diqqət), sonra isə hesablama (İrəli Bəsləmə).
    """
    def __init__(self, yerlesdirme_olcusu, bash_sayi):
        super().__init__()
        bash_olcusu = yerlesdirme_olcusu // bash_sayi
        self.diqqet = ChoxBashliDiqqet(bash_sayi, bash_olcusu)
        self.hesablama = IreliBesleme(yerlesdirme_olcusu)
        self.norma1 = nn.LayerNorm(yerlesdirme_olcusu) # Məlumatları stabilləşdirir
        self.norma2 = nn.LayerNorm(yerlesdirme_olcusu)

    def forward(self, x):
        # Qalıq bağlantılar (Residual connections) vasitəsilə məlumatın itməsinin qarşısını alırıq
        x = x + self.diqqet(self.norma1(x))
        x = x + self.hesablama(self.norma2(x))
        return x

class IsmayilModeli(nn.Module):
    """ İsmayılın əsas Dil Modeli (Language Model) memarlığı. """
    def __init__(self, luget_olcusu):
        super().__init__()
        # Simvolların rəqəmsal qarşılığı (Token Embeddings)
        self.simvol_cedveli = nn.Embedding(luget_olcusu, yerlesdirme_olcusu)
        # Simvolların mətndəki mövqeyi (Positional Embeddings)
        self.movqe_cedveli = nn.Embedding(blok_olcusu, yerlesdirme_olcusu)
        # Arxa-arxaya düzülmüş Transformer blokları
        self.bloklar = nn.Sequential(*[TransformerBloku(yerlesdirme_olcusu, bash_sayi=bash_sayi) for _ in range(lay_sayi)])
        self.son_norma = nn.LayerNorm(yerlesdirme_olcusu)
        # Rəqəmləri yenidən simvolların ehtimallarına çevirən qat
        self.bash_qati = nn.Linear(yerlesdirme_olcusu, luget_olcusu)

    def forward(self, indeksler, hedefler=None):
        B, T = indeksler.shape
        
        simvol_embs = self.simvol_cedveli(indeksler) # (Batch, Time, Channel)
        movqe_embs = self.movqe_cedveli(torch.arange(T, device=indeksler.device)) # (Time, Channel)
        
        # Məumat və mövqe enerjisini birləşdiririk
        x = simvol_embs + movqe_embs 
        x = self.bloklar(x)    
        x = self.son_norma(x)      
        ehtimallar = self.bash_qati(x) # (B, T, luget_olcusu)

        if hedefler is None:
            itki = None
        else:
            B, T, C = ehtimallar.shape
            ehtimallar = ehtimallar.view(B*T, C)
            hedefler = hedefler.view(B*T)
            # Modelin səhvini (itkisini) hesablayırıq
            itki = F.cross_entropy(ehtimallar, hedefler)

        return ehtimallar, itki

    def yeni_metn_yarat(self, indeksler, maksimum_yeni_simvol):
        # Verilmiş başlanğıc mətni əsasında yeni simvollar generasya edir
        for _ in range(maksimum_yeni_simvol):
            # Konteksti blok ölçüsünə uyğun kəsirik
            indeks_kontekst = indeksler[:, -blok_olcusu:]
            ehtimallar, itki = self(indeks_kontekst)
            # Yalnız ən sonuncu simvolun ehtimalına baxırıq
            ehtimallar = ehtimallar[:, -1, :] 
            yumshaq_ehtimal = F.softmax(ehtimallar, dim=-1)
            # Ehtimallara uyğun təsadüfi növbəti simvolu seçirik
            novbeti_indeks = torch.multinomial(yumshaq_ehtimal, num_samples=1) 
            # Yeni simvolu mövcud mətnin sonuna əlavə edirik
            indeksler = torch.cat((indeksler, novbeti_indeks), dim=1) 
        return indeksler
