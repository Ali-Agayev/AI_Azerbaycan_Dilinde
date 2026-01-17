import torch

def weights_yeniden_adlandir():
    yol = 'ismayil_model.pth'
    print(f"'{yol}' faylındakı çəkilər yenidən adlandırılır...")
    
    # Köhnə çəkiləri yükləyirik
    kok_ceki = torch.load(yol, map_location='cpu')
    yeni_ceki = {}
    
    # Köhnə adlardan yeni Azərbaycan adlarına xəritə
    ad_xeritesi = {
        'token_embedding_table': 'simvol_cedveli',
        'position_embedding_table': 'movqe_cedveli',
        'blocks': 'bloklar',
        'sa.heads': 'diqqet.bashlar',
        'sa.proj': 'diqqet.proyeksiya',
        'sa.dropout': 'diqqet.seyriltme',
        'ffwd.net.0': 'hesablama.shabaka.0',
        'ffwd.net.2': 'hesablama.shabaka.2',
        'ffwd.net.3': 'hesablama.shabaka.3',
        'ln1': 'norma1',
        'ln2': 'norma2',
        'ln_f': 'son_norma',
        'lm_head': 'bash_qati'
    }
    
    for kohne_ad, deyer in kok_ceki.items():
        yeni_ad = kohne_ad
        for ingilisce, azərbaycanca in ad_xeritesi.items():
            yeni_ad = yeni_ad.replace(ingilisce, azərbaycanca)
        
        yeni_ceki[yeni_ad] = deyer
        print(f"Dəyişdirildi: {kohne_ad} -> {yeni_ad}")
    
    # Yeni çəkiləri eyni fayla və ya yeni fayla yazırıq
    torch.save(yeni_ceki, yol)
    print("Bütün çəkilər uğurla yenidən adlandırıldı!")

if __name__ == "__main__":
    weights_yeniden_adlandir()
