from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import torch
import os
from model import IsmayilModeli # Bizim yeni model sinfi
from tokenizer import CharTokenizator # Bizim yeni tokenizator sinfi

# FastAPI tətbiqini yaradırıq
ismayil_server = FastAPI(title="İsmayılın Şəxsi AI Serveri")

# CORS tənzimləmələri (Frontendin backend ilə danışmasına icazə vermək üçün)
ismayil_server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Bütün mənbələrə icazə veririk
    allow_credentials=True,
    allow_methods=["*"], # Bütün metodlara (GET, POST və s.) icazə veririk
    allow_headers=["*"],
)

# Qlobal dəyişənlər - Model və Tokenizatoru burada saxlayırıq ki, hər dəfə yükləməyək
cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizator = None
ismayil_modeli = None

def ai_ni_bashlat():
    """ 
    Modeli və Tokenizatoru yaddaşdan yükləyən funksiya.
    Əgər model təlim olunmayıbsa, False qaytarır.
    """
    global tokenizator, ismayil_modeli
    if ismayil_modeli is None:
        # Faylların varlığını yoxlayırıq
        if not os.path.exists('tokenizer.json') or not os.path.exists('ismayil_model.pth'):
            return False
            
        # Tokenizatoru yükləyirik
        tokenizator = CharTokenizator.yukle('tokenizer.json')
        # Model memarlığını yaradırıq
        ismayil_modeli = IsmayilModeli(tokenizator.luget_olcusu)
        # Öyrənilmiş çəkiləri (weights) modelə yükləyirik
        ismayil_modeli.load_state_dict(torch.load('ismayil_model.pth', map_location=cihaz))
        ismayil_modeli.to(cihaz)
        ismayil_modeli.eval() # Modeli yalnız cavab vermə (inference) rejiminə salırıq
        print("İsmayıl AI uğurla işə düşdü və suallarınızı gözləyir!")
    return True

# API istəkləri üçün məlumat strukturları (Pydantic modelləri)
class Mesaj(BaseModel):
    role: str # 'user' və ya 'assistant'
    content: str # Mesajın mətni

class ChatIsteyi(BaseModel):
    messages: List[Mesaj]

@ismayil_server.get("/")
async def ana_sehife():
    # Serverin aktiv olub-olmadığını yoxlamaq üçün kiçik endpoint
    return {"status": "online", "model": "İsmayıl Custom Transformer"}

@ismayil_server.post("/v1/chat/completions")
async def chat_cavabi(istek: ChatIsteyi):
    """
    Bu əsas hissədir. İstifadəçidən gələn mesajı AI-yə göndərir və cavab alır.
    """
    # AI-nin hazır olub-olmadığını yoxlayırıq
    if not ai_ni_bashlat():
        raise HTTPException(status_code=503, detail="İsmayıl hələ təlim keçir və ya model tapılmadı.")
        
    try:
        # İstifadəçinin son göndərdiyi sualı götürürük
        son_mesaj = istek.messages[-1].content
        
        # Mətni rəqəmlərə (tokenlərə) çeviririk
        giriş_idləri = torch.tensor([tokenizator.kodlasdir(son_mesaj)], dtype=torch.long, device=cihaz)
        
        # AI-dən yeni simvollar generasya etməsini istəyirik
        with torch.no_grad():
            # generate() yerinə yeni_metn_yarat() istifadə edirik (model.py-da adını dəyişmişik)
            butun_idlar = ismayil_modeli.yeni_metn_yarat(giriş_idləri, maksimum_yeni_simvol=100)
            # Yalnız AI-nin yaratdığı hissəni kəsib götürürük (giriş mətnini çıxarıq)
            cavab_idlari = butun_idlar[0][len(giriş_idləri[0]):].tolist()
            # Rəqəmləri yenidən başa düşülən mətnə çeviririk
            cavab_metni = tokenizator.de_kodlasdir(cavab_idlari)
            
        # OpenAI formatına uyğun cavab qaytarırıq (Frontend bunu gözləyir)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": cavab_metni
                    }
                }
            ]
        }
    except Exception as xata:
        raise HTTPException(status_code=500, detail=str(xata))

if __name__ == "__main__":
    # Serveri 8000-ci portda başladırıq
    uvicorn.run(ismayil_server, host="0.0.0.0", port=8000)
