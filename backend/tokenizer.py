import json

class CharTokenizator:
    """
    Bu sinif mətnləri ayrı-ayrı simvollar (hərflər, rəqəmlər) səviyyəsində rəqəmlərə (tokenlərə) 
    çevirmək və əksinə, rəqəmləri mətnə qaytarmaq üçün istifadə olunur.
    """
    def __init__(self, metn=""):
        # Əgər mətn verilibsə, ondan unikal simvollar siyahısını (vocabı) yaradırıq
        if metn:
            self.simvollar = sorted(list(set(metn)))
        else:
            # Standart olaraq istifadə olunacaq simvollar dəsti
            xususi_simvollar = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n"
            self.simvollar = sorted(list(set(xususi_simvollar)))
        
        # Lüğətin ümumi ölçüsü
        self.luget_olcusu = len(self.simvollar)
        
        # Simvoldan rəqəmə (String to Integer) çevirmə xəritəsi
        self.simvoldan_reqeme = { simvol:indeks for indeks, simvol in enumerate(self.simvollar) }
        
        # Rəqəmdən simvola (Integer to String) çevirmə xəritəsi
        self.reqemden_simvola = { indeks:simvol for indeks, simvol in enumerate(self.simvollar) }

    def kodlasdir(self, s):
        # Mətni rəqəmlər siyahısına çevirir
        return [self.simvoldan_reqeme[c] for c in s if c in self.simvoldan_reqeme]

    def de_kodlasdir(self, l):
        # rəqəmlər siyahısını yenidən mətnə çevirir
        return ''.join([self.reqemden_simvola[i] for i in l])

    def yadda_saxla(self, yol):
        # Tokenizatorun simvollarını JSON faylı kimi yadda saxlayır
        with open(yol, 'w', encoding='utf-8') as f:
            json.dump(self.simvollar, f)

    @classmethod
    def yukle(cls, yol):
        # Yadda saxlanılmış JSON faylından tokenizatoru yenidən yükləyir
        with open(yol, 'r', encoding='utf-8') as f:
            hazir_simvollar = json.load(f)
        obyekt = cls()
        obyekt.simvollar = hazir_simvollar
        obyekt.luget_olcusu = len(hazir_simvollar)
        obyekt.simvoldan_reqeme = { s:i for i, s in enumerate(hazir_simvollar) }
        obyekt.reqemden_simvola = { i:s for i, s in enumerate(hazir_simvollar) }
        return obyekt
