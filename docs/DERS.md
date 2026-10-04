# Ders: Bir Maymun Beynine Karar Vermeyi Öğretmek

Bu ders, projedeki her parçanın **neden** öyle yapıldığını sırayla anlatır. Her bölüm bir
komutla ve bir kod dosyasıyla eşleşir. Okurken komutları çalıştır ve şekillere bak.

---

## Bölüm 0: Büyük resim

Profesyonel bir hesaplamalı sinirbilim projesi dört katmandan oluşur:

1. **Biyoloji:** Gerçek beyinde ne biliyoruz? (deneyler, ölçümler)
2. **Model:** Bunu hangi denklemlerle temsil ediyoruz? (nöron, sinaps, ağ)
3. **Öğrenme:** Ağ deneyimden nasıl değişir? (ML veya biyolojik kural)
4. **Doğrulama:** Model gerçek beyne benziyor mu? (aynı testleri uygula, karşılaştır)

En sık yapılan hata, 4. adımı atlamaktır. "Ağım %99 doğru" demek hiçbir şey ifade etmez.
Asıl soru şu: **Ağ, maymunun yaptığı hataları da yapıyor mu?**

Bu projedeki zincir:

```
Gerçek deney (Newsome, Shadlen)  ->  config.py (parametreler)  ->  task.py + network.py (model)
      ->  learning/ (iki öğrenme kuralı)  ->  analysis.py (maymunla aynı ölçümler)  ->  plots.py
```

---

## Bölüm 1: Nöron ve spike

**Komut:** `python -m monkeybrain neuron` · **Kod:** `monkeybrain/neurons.py`

Bir nöron, elektrik yükünü biriktiren sızdıran bir kaptır. Membran potansiyeli `v` girdiyle
yükselir, zamanla dinlenme değerine geri sızar. Bir eşiği geçince nöron **spike** atar
(yaklaşık 1 ms süren bir elektrik darbesi) ve sıfırlanır. Beyindeki bilgi bu spike'larla taşınır.

### Adaptif LIF (ağda kullandığımız model)

```
τ_m · dv/dt = −v + I(t)                (sızıntı + girdi)
v ≥ v_th + β·a   ise:  spike, v ← 0    (eşik)
τ_a · da/dt = −a,  her spike'ta a ← a + 1   (adaptasyon)
```

- `τ_m` (membran zaman sabiti): piramidal hücrelerde ~20 ms, hızlı internöronlarda ~10 ms.
- **Adaptasyon:** Piramidal hücreler sürekli uyarılınca yavaşlar. Her spike eşiği biraz
  yükseltir. Şekildeki ALIF panelinde spike aralıklarının uzadığını gör.
- **Refrakter dönem:** Spike'tan sonra 2 ms boyunca nöron ateşleyemez. Bu yüzden hız asla
  sonsuz olmaz.

Normalize birim kullanıyoruz: `v = 0` dinlenme (~−70 mV), `v = 1` eşik (~−50 mV).

### Izhikevich modeli (yalnızca demo)

Izhikevich (2003), dört parametreyle (a, b, c, d) neokortekste gözlenen ateşleme desenlerini
üreten bir model önerdi. Şekilde aynı akım beş farklı hücreye veriliyor:

- **RS** (düzenli ateşleyen): tipik piramidal hücre, adapte olur
- **IB** (içsel patlamalı): önce bir patlama, sonra düzenli
- **CH** (chattering): hızlı tekrarlayan patlamalar
- **FS** (hızlı ateşleyen): baskılayıcı internöron, adaptasyon yok
- **LTS** (düşük eşikli): başka bir internöron tipi

Ağda neden Izhikevich değil de ALIF kullanıyoruz? ALIF daha basit ve daha hızlı. En önemlisi,
PyTorch ile geri yayılıma (Bölüm 4) daha uygun.

> **Kendin dene:** `neurons.py` içinde `IZHIKEVICH_TYPES` sözlüğüne Izhikevich'in
> "rezonatör" parametrelerini (a=0.1, b=0.26, c=−60, d=−1) ekle ve komutu tekrar çalıştır.

---

## Bölüm 2: Makak korteksi, MT ve LIP

**Komut:** `python -m monkeybrain simulate` · **Kod:** `monkeybrain/network.py`, `config.py`

### İki beyin bölgesi

- **MT (V5):** Hareket bölgesi. Her MT nöronunun tercih ettiği bir yön vardır. Noktalar o
  yöne giderse daha çok ateşler. Newsome ve meslektaşları, MT'yi elektrikle uyarınca
  maymunun kararının değiştiğini gösterdi: MT, hareket algısının kaynağıdır.
- **LIP:** Parietal korteks. Shadlen ve Newsome (2001), LIP nöronlarının kanıtı zamanla
  **biriktirdiğini** gösterdi. Maymun sağa bakmaya karar verecekse, sağ tarafı temsil eden
  LIP nöronlarının aktivitesi yavaşça yükselir (rampa).

### Ağın biyolojik kuralları (hepsi `config.py`'de, gerekçeleriyle)

| Kural | Neden? | Kodda |
|---|---|---|
| %80 uyarıcı, %20 baskılayıcı | Korteksteki gözlenen oran | `n_exc=160, n_inh=40` |
| **Dale yasası** | Bir nöron ya hep uyarır ya hep baskılar, asla ikisini birden yapmaz | `sign` vektörü + `apply_constraints()` |
| Seyrek bağlantı (%20) | Lokal kortekste her nöron diğerlerinin küçük bir kısmına bağlıdır | `p_rec=0.2` |
| Yavaş NMDA sinapsları (E→E) | Wang (2002): kanıt biriktirmeyi sağlayan yavaş yankılanma | `tau_nmda=100` |
| NMDA doyumu | Reseptörlerin hepsi açılınca akım artık büyümez | `nmda_kappa=0.5` |
| Hızlı AMPA/GABA sinapsları | Diğer bağlantılar ~5 ms | `tau_syn=5` |
| Arka plan gürültüsü | Korteksin geri kalanından gelen düzensiz girdi | `bg_mean`, `bg_std` |

### İki seçim havuzu ve rekabet

LIP'teki E nöronlarının 40'ı "Sol", 40'ı "Sağ" havuzudur. Her havuz kendini güçlü uyarır
(`w_plus`), ama iki havuz ortak bir baskılayıcı popülasyonu paylaşır. Biri yükselince I
nöronları da yükselir ve diğerini bastırır. Buna **kazanan-hepsini-alır** denir. Wang (2002)
modelinin temel fikri budur.

### Mühendislik dersi: sabit girdi derecesi

İlk denemede bağlantılar tamamen rastgeleydi. Sonuç: %0 tutarlılıkta bile denemelerin %95'inde
Sol havuz kazandı. Neden? Rastgele bağlantı sayıları havuzlar arasında ~%5 fark yarattı ve
güçlü özyineleme bu küçük farkı büyüttü. Çözüm (Brunel 2000): her nöron, her gruptan **tam
olarak** aynı sayıda girdi alır (`_fixed_indegree`).
Bedeli: ağ çok homojen. `rates_untrained.png`'de hızlar birkaç sütunda toplanıyor; gerçek
korteksin geniş (log-normal) dağılımı yok. Bu bilinçli bir basitleştirme.

### Mühendislik dersi: NMDA doyumu ve gerçekçi hızlar

İlk sürümde NMDA akımı doğrusaldı: her spike akımı aynı miktarda artırıyordu. Sonuç: kazanan
havuz deneme bitene kadar durmadan tırmandı ve **113 Hz**'e ulaştı. Gerçek LIP nöronları kararda
~60–70 Hz civarına çıkar.

Gerçek NMDA reseptörleri **doyuma ulaşır**. Bir sinapstaki reseptörlerin hepsi açıldıktan sonra
daha fazla spike akımı artıramaz. Modelde her E nöronu için 0 ile 1 arasında bir "açık oran" `g`
tutuluyor:

```
g <- g · e^(−dt/τ_NMDA) + κ · (1 − g) · spike
```

`(1 − g)` çarpanı yüzünden `g` asla 1'i geçemez. Bu yüzden özyinelemeli uyarım sınırlıdır, ve
kazanan havuz sonlu bir hızda **platoya** ulaşır (Wang 2002 modelindeki "çekici durum").
Doyum özyinelemeyi zayıflattığı için `w_plus` (24 → 48) ve ortak baskılama (`w_ei`, `w_ie`)
yeniden ayarlandı. Sonuç: biyolojik modelin tepe hızı 113 → 42 Hz.

> **Kendin dene:** `config.py`'de `w_plus`'ı 48'den 20'ye düşür ve `simulate` çalıştır.
> Raster'da havuzların artık yarışmadığını göreceksin.

---

## Bölüm 3: Deney (rastgele nokta görevi)

**Kod:** `monkeybrain/task.py`

Gerçek deneyde maymun bir noktaya bakar (fiksasyon, 100 ms). Sonra ekranda hareketli
noktalar belirir (600 ms). Noktaların bir kısmı aynı yöne gider, gerisi rastgeledir.

- **Tutarlılık (coherence):** Aynı yöne giden noktaların oranı. %0'da hiç sinyal yoktur.
  %51.2'de yarısından fazlası aynı yöne gider. Kullanılan seviyeler laboratuvardakilerle
  aynı: 0, 3.2, 6.4, 12.8, 25.6, 51.2.
- **Ödül:** Doğru cevapta meyve suyu. %0'da cevap rastgele ödüllendirilir.

### MT nasıl modellendi?

```
r_i(t) = r_taban + g · c(t) · cos(θ_uyaran − θ_tercih_i)
```

- Hareket yokken MT ~8 Hz ateşler. Hareket başlayınca %0 tutarlılıkta bile ~20 Hz'e çıkar.
- Tutarlılık arttıkça tercih edilen yöndeki hız doğrusal artar (Britten et al. 1993).
- `c(t)` her ekran karesinde (16 ms) **dalgalanır**. Rastgele noktaların anlık hareketi
  gürültülüdür; görevi zorlaştıran şey budur. Bu gürültü olmasaydı ağ %3.2'de bile kusursuz
  olurdu, gerçek maymun ise değildir.
- MT spike'ları Poisson sürecidir: her ms'de olasılık `r · dt`.

---

## Bölüm 4: ML ile öğrenme (surrogate gradient)

**Komut:** `python -m monkeybrain train --method ml` · **Kod:** `learning/surrogate_trainer.py`

### Problem: spike türevlenemez

Derin öğrenme, kaybın her ağırlığa göre türevini alarak çalışır (geri yayılım). Spike ise bir
basamak fonksiyonu: eşiğin altında 0, üstünde 1. Türevi her yerde 0, eşikte sonsuz. Gradyan
akmaz.

### Çözüm: "vekil" (surrogate) türev

İleri yönde gerçek basamağı kullan, geri yönde yumuşak bir yaklaşım kullan
(Neftci, Mostafa & Zenke 2019):

```
ileri:  s = 1  eğer v > eşik, aksi halde 0
geri:   ds/dv ≈ 1 / (10·|v − eşik| + 1)²       ("hızlı sigmoid" türevi)
```

Kodda `SurrogateSpike` sınıfı tam olarak bunu yapar (`neurons.py`).

### Eğitim döngüsü

1. 32 deneme üret (rastgele tutarlılık ve yön).
2. Ağı 700 adım (700 ms) simüle et.
3. Son 200 ms'deki havuz hızlarından karar çıkar → **çapraz entropi kaybı**.
4. **Hız düzenlileştirme:** Bir nöronun ortalaması 50 Hz'i aşarsa veya 1 Hz'in altına düşerse ceza.
   Bu olmadan ağ biyolojik olmayan 200 Hz çözümler bulabilir.
5. Hatayı 700 adım boyunca geriye yay (BPTT) ve Adam ile **tüm** ağırlıkları güncelle.
6. `apply_constraints()`: Dale yasasını koru.

### Mühendislik dersi: performans tuzağı

İlk sürümde girdiyi `mt_input[t]` diye dilimliyorduk. Her adım 10 saniye sürdü. Neden?
PyTorch'ta büyük bir tensörü 700 kez dilimlemek, geri yayılımda 700 kez tam boyutlu sıfır
tensörü oluşturur. `unbind(0)` ile adım ~1 saniyeye indi. Ders: **önce ölç, sonra optimize et.**

### Neden "beyin böyle öğrenmez"?

BPTT için bir sinapsın, 500 ms önce başka bir nöronda olan şeyin hatadaki payını "bilmesi"
gerekir. Biyolojik sinapsların böyle bir bilgiye erişimi yoktur. ML yöntemi bize **ne
öğrenilebileceğinin** üst sınırını gösterir, beynin **nasıl** öğrendiğini değil.

---

## Bölüm 5: Beyin gibi öğrenme (dopamin + STDP)

**Komut:** `python -m monkeybrain train --method bio` · **Kod:** `learning/rstdp_trainer.py`

### Üç faktörlü kural

1. **Önceki nöron** spike atar.
2. **Sonraki nöron** spike atar. **STDP** (Bi & Poo 1998): önce→sonra sırası (yaklaşık 20 ms içinde)
   sinapsı güçlendirir, sonra→önce zayıflatır. Ama ağırlık hemen değişmez; sinapsta bir
   **uygunluk izi** (eligibility trace) kalır. Bu iz ~800 ms'de söner.
3. **Dopamin:** Deneme bitince maymun meyve suyunu alır ya da almaz. Dopamin nöronları
   **ödül tahmin hatasını** kodlar (Schultz, Dayan & Montague 1997):
   ```
   D = alınan ödül − beklenen ödül
   ```
   Beklenen ödül her zorluk seviyesi için ayrı öğrenilir ("eleştirmen"). Kolay denemede
   doğru cevap sürpriz değildir (D ≈ 0); zor denemede doğru cevap büyük bir sürprizdir (D > 0).

```
ΔW = η · D · e        (Izhikevich 2007; Frémaux & Gerstner 2016)
```

### Mühendislik dersi: kredi atama sorunu

İlk sürüm öğrenmedi (%52). Teşhis: dopamin **tüm** sinapslara aynı anda gider. Kaybeden havuz
da düşük hızda ateşlemeye devam ettiği için, kazanan havuzun ödülünden o da pay aldı. İki havuz
da aynı yönü öğrendi.

İki düzeltme yapıldı:

1. **EH kuralı** (Legenstein et al. 2010): Ham spike yerine, nöronun **her zamankinden sapması**
   kullanılır (`post_centered = spike − ortalama`). Her zamankinden fazla ateşleyen nöron kredi
   alır. Bu kural, maymunların beyin-bilgisayar arayüzü öğrenmesini açıklamak için önerildi.
2. **Güçlü rekabet (attractor):** Havuzlar arası rekabet güçlendirildi (NMDA, `w_plus`).
   Kaybeden havuz susturulunca uygunluk izi neredeyse yalnızca kazanan havuzda birikir.

Bu iki adımdan sonra biyolojik yöntem de öğrendi. Gerçek beyinde de öğrenme kuralı ile
devrenin yapısı birlikte çalışır.

### Sinaptik ölçekleme

Her nöronun toplam girdi ağırlığı sabit tutulur. Bir sinaps güçlenirse diğerleri orantılı
zayıflar. Böylece ağırlıklar sonsuza büyümez ve girdiler birbiriyle yarışır.

> **Kendin dene:** `rstdp_trainer.py`'de `post_centered` yerine ham `post` kullan
> ve `--quick` ile eğit. Öğrenmenin bozulduğunu göreceksin.

---

## Bölüm 6: Sonuçları okumak (model maymuna benziyor mu?)

**Komut:** `python -m monkeybrain evaluate` ve `compare` · **Kod:** `analysis.py`

### Psikometrik eğri (`psychometric.png`)

X ekseni tutarlılık, Y ekseni doğruluk. Weibull fonksiyonu uydurulur:

```
P(doğru) = 1 − 0.5 · exp(−(c/α)^β)
```

**α (eşik)**, doğruluğun ~%82'ye ulaştığı tutarlılıktır. Küçük α, daha hassas bir gözlemci
demektir. Grafikteki kesikli çizgi, tipik maymun performansını temsil eden **yaklaşık** bir
referanstır (α ≈ %10). Gerçek veri değildir; farklı maymunlarda ve deneylerde değişir.

### LIP rampası (`lip_ramp.png`): en önemli şekil

Roitman & Shadlen (2002)'nin ünlü bulgusu:
- Seçilen yöndeki LIP nöronlarının aktivitesi zamanla yükselir (rampa).
- **Tutarlılık arttıkça rampa daha diktir.** Güçlü kanıt, daha hızlı birikim demektir.
- Seçilmeyen yöndeki nöronlar bastırılır.

Ağımıza bu davranışı **açıkça öğretmedik**. Yalnızca doğru cevabı ödüllendirdik. Rampa
kendiliğinden ortaya çıkarsa, model maymunun beynindeki mekanizmaya benziyor demektir.

### Kronometrik eğri (`chronometric.png`)

Zor kararlar daha uzun sürmeli. Karar süresi, bir havuzun bir eşik hızı (Hz) ilk geçtiği an
olarak ölçülür. Maymunun gerçek tepki süresi buna ~300 ms motor gecikmesi ekler.

**Ölçüm tuzağı:** İlk sürümde eşik yüksekti (~26 Hz). Zor denemelerin çoğu 600 ms içinde eşiğe
hiç ulaşmadı; ortalama yalnızca eşiği geçen birkaç "şanslı" denemeden hesaplandı ve eğri
anlamsız çıktı (hayatta kalan yanlılığı). Şimdi eşik otomatik seçiliyor: **en zor koşulda bile
denemelerin %80'inin ulaştığı en yüksek hız** (`analysis.decision_threshold`).

### Ağırlıklar (`weights.png`)

X ekseni MT nöronunun tercih yönü. Eğitimden önce her şey düz. Eğitimden sonra:
- Sağ havuz (turuncu), 0° (sağ) tercih eden MT nöronlarından güçlü girdi almalı.
- Sol havuz (mavi), 180° (sol) tercih edenlerden.

Ağ, **hangi MT nöronunu dinleyeceğini** öğrenmiş olur.

### Bu projede elde edilen sonuçlar (seed 42)

| Tutarlılık | ML | Biyolojik | Maymun (yaklaşık) |
|---|---|---|---|
| %0 | %47.0 | %44.5 | %50 |
| %3.2 | %62.0 | %54.0 | %60 |
| %6.4 | %76.0 | %64.5 | %71 |
| %12.8 | %86.0 | %76.5 | %87 |
| %25.6 | %99.0 | %93.5 | %98 |
| %51.2 | %100 | %100 | %100 |
| **Psikometrik eşik** | **%9.4** | **%15.3** | **~%10** |
| **Seçilen havuzun tepe hızı** | **30 Hz** | **42 Hz** | **~60–70 Hz** |

Her koşulda 200 deneme var, yani her doğruluk değerinin ±%3.5 civarında örnekleme
belirsizliği var. %0'da doğru cevap yoktur. Biyolojik model bu koşulda %50.5 "sağ" diyor,
yani yanlılık yok. ML modeli %40 "sağ" diyor; bu hafif bir sol yanlılık (~3 standart hata).

**Ne gördük?**

1. **İki yöntem de öğrendi.** ML'nin psikometrik eğrisi maymun referansıyla neredeyse üst
   üste (eşik %9.4). Biyolojik model daha az hassas (eşik %15.3).
2. **İki yöntem de ~1000 denemede hızla öğrendi, sonra yavaşça iyileşti.** R-STDP yalnızca
   5.120 MT → LIP(E) sinapsını değiştiriyor ve hazır bir rekabet devresinin üstünde çalışıyor.
   ML ise 14.400 sinapsın (MT → LIP ve LIP içi) hepsini birden ayarlıyor.
   Dikkat: Gerçek maymunlar bu görevi aylarca öğrenir; iki model de öğrenme **süresinin**
   bir modeli değildir.
3. **LIP rampası iki modelde de kendiliğinden ortaya çıktı.** Tutarlılık arttıkça rampa
   dikleşiyor, seçilmeyen havuz bastırılıyor. Biyolojik modelde rampa ~30–35 Hz'de platoya
   ulaşıyor ve kolay denemelerde oraya daha erken varıyor. Bu da Roitman & Shadlen'in tepki
   süresi deneyine benzer: kararda LIP hızları ortak bir seviyede buluşur.
4. **Tepe hızları artık biyolojik aralıkta** (Bölüm 2, NMDA doyumu). İki model de gerçek
   LIP'in tepe hızının (~60–70 Hz) biraz altında kalıyor. İlk sürümde biyolojik model 113 Hz'e
   çıkıyordu; düzeltmenin bedeli biraz daha yüksek eşik oldu (%12.5 → %15.3).
5. **Karar süresi iki modelde de tutarlılıkla kısalıyor.** Düşük tutarlılıkta düz, yüksek
   tutarlılıkta hızla düşüyor; bu şekil maymunlarda da görülür. İki modelin mutlak süreleri
   doğrudan kıyaslanamaz, çünkü eşik her model için ayrı hesaplanıyor.
6. **Ağırlık profilleri neredeyse aynı:** İki yöntem de birbirinden bağımsız olarak aynı
   çözümü buldu: sağ havuz sağı, sol havuz solu tercih eden MT nöronlarını dinliyor (kosinüs
   biçimli profil).

---

## Bölüm 6b: 3D beyin (`python -m monkeybrain brain3d`)

**Kod:** `monkeybrain/export3d.py` (veri) ve `monkeybrain/templates/brain3d.html` (Three.js)

Bu komut, eğitilmiş modelleri gerçek simülasyon verisiyle etkileşimli bir 3D sayfaya dönüştürür.

- **Beyin:** Şematik bir makak yarıküresi. Gerçek sulkuslar (lateral, STS, intraparietal, santral,
  arkuat, prensipal, lunat) ve bölgeler: V1, MT, LIP, FEF, VTA. Konumlar şematiktir.
- **Lupa kümeleri:** MT ve LIP nöronları büyütülmüş olarak gösterilir. MT halkasında her nöronun
  konumu ve rengi tercih ettiği yönü gösterir; halkanın sağı sağa duyarlıdır.
- **Spike akışı:** Her MT spike'ı beyaz cevher yolu boyunca LIP'e giden bir ışık parçacığıdır.
  LIP nöronları ateşlendikçe parlar (kalsiyum görüntülemedeki gibi).
- **Öğrenme aşamaları:** Eğitim sırasında ağırlıkların anlık görüntüleri kaydedilir (eğitimin
  %0, 2, 5, 15, 40 ve 100'ünde). Her aşamada **aynı uyaran ve aynı gürültü** ağa verilir; tek
  fark öğrenilen ağırlıklardır. Aşama değiştirince bağlantıların nasıl parladığını izle.
  Başlangıçta hepsi soluk, sonda sağa duyarlı MT nöronlarından sağ havuza parlak bir demet var.
- **Deneme sonu:** Biyolojik modda VTA'dan altın renkli bir dopamin dalgası yayılır, ve bu
  denemede uygunluk izi taşıyan sinapslar parlar. ML modunda pembe hata sinyali bağlantılar
  boyunca geriye akar.

URL parametreleriyle belirli bir anı açabilirsin, ör. `beyin_3d.html?method=ml&stage=0&t=450&pause&view=LIP`.
Diğer parametreler: `lang=tr` / `lang=en` (dil), `clean` (yalnızca sahne; sunum ve ekran kaydı için),
`post=1000` (deneme sonundaki dopamin / hata anına git).

Canlı sürüm: <https://ethyusuf.github.io/macaque-decision-snn/>

### Bulutta çalıştırmak (Kaggle / Colab)

`notebooks/macaque-decision-snn.ipynb`, her şeyi baştan yapar: testler, iki eğitim, maymun testleri ve
3D sayfa. README'deki "Open in Kaggle" düğmesiyle açılır. Kaggle'da *Settings → Internet* açık olmalı.
Notebook `notebooks/build_notebook.py` ile üretilir; hücreleri değiştirmek için bu dosyayı düzenle.

### Hazır modeller ve İngilizce şekiller

Depoda `pretrained/ml` ve `pretrained/bio` altında eğitilmiş modeller var. Henüz eğitim yapmadıysan
`compare`, `evaluate` ve `brain3d` bunları otomatik kullanır. Şekilleri İngilizce üretmek için komuttan
önce `--lang en` yaz, ör. `python -m monkeybrain --lang en compare`.

---

## Bölüm 7: Sınırlamalar ve sonraki adımlar

### Bu model ne DEĞİL?

- **Gerçek bir maymun beyni değil.** 264 nöronluk bir model; gerçek LIP'te milyonlarca nöron var.
- **V1 yok.** Hareket sinyali doğrudan MT'ye veriliyor.
- **Tek bir karar kuralı var.** Gerçek maymun ne zaman karar vereceğine kendisi karar verir
  (tepki süresi görevi). Bizde uyaran süresi sabit, karar süresi sonradan ölçülüyor.
- **Ağ çok homojen** (Bölüm 2). Gerçek nöronlar çok daha çeşitlidir.
- **Parametrelerin bir kısmı "model seçimi".** `config.py`'deki her parametrenin yanında
  bu açıkça yazıyor. Gerçek bir araştırmada bu parametreler verilere uydurulur.

### Profesyonel bir sonraki adım ne olurdu?

1. **Gerçek veriyle karşılaştırma:** DANDI arşivinde gerçek makak kayıtları var (ör. Neural
   Latents Benchmark). Model nöronlarının hızlarını gerçek nöronlarla karşılaştırmak
   (ör. temsil benzerlik analizi) bir sonraki doğrulama katmanı olurdu.
2. **Tepki süresi görevi:** Ağ eşiğe ulaşınca denemeyi bitir; maymunun tepki süresi dağılımıyla
   karşılaştır.
3. **Daha büyük ve heterojen ağ:** GPU ile binlerce nöron, log-normal ağırlıklar.
4. **Diğer biyolojik öğrenme kuralları:** e-prop (Bellec et al. 2020), BPTT'nin biyolojik
   olarak makul bir yaklaşımıdır. Üçüncü bir yöntem olarak eklenebilir.

---

## Kaynaklar

- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *J Neurosci*.
- Britten, K. H., Shadlen, M. N., Newsome, W. T. & Movshon, J. A. (1993). Responses of neurons in macaque MT to stochastic motion signals. *Visual Neuroscience*.
- Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. *J Comput Neurosci*.
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated STDP and theory of three-factor learning rules. *Front Neural Circuits*.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Trans Neural Networks*.
- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex*.
- Legenstein, R., Chase, S. M., Schwartz, A. B. & Maass, W. (2010). A reward-modulated Hebbian learning rule can explain experimentally observed network reorganization in a brain control task. *J Neurosci*.
- Neftci, E. O., Mostafa, H. & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*.
- Roitman, J. D. & Shadlen, M. N. (2002). Response of neurons in the lateral intraparietal area during a combined visual discrimination reaction time task. *J Neurosci*.
- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*.
- Shadlen, M. N. & Newsome, W. T. (2001). Neural basis of a perceptual decision in the parietal cortex (area LIP) of the rhesus monkey. *J Neurophysiol*.
- Wang, X.-J. (2002). Probabilistic decision making by slow reverberation in cortical circuits. *Neuron*.
