"""Tüm parametreler tek yerde.

Her parametrenin yanında ya biyolojik gerekçesi ya da "model seçimi" notu var.
"Model seçimi" = literatürde kesin bir karşılığı yok; simülasyonun düzgün
çalışması için seçildi. Membran potansiyeli normalize birimdedir:
v = 0 dinlenme potansiyeli (~ -70 mV), v = 1 ateşleme eşiği (~ -50 mV).
"""

from dataclasses import asdict, dataclass, field


@dataclass
class TaskParams:
    """Rastgele nokta hareket görevi (Newsome, Britten & Shadlen deneyleri)."""

    dt: float = 1.0                 # ms, simülasyon adımı
    t_fix: int = 100                # ms, fiksasyon: ekranda nokta var, hareket yok
    t_stim: int = 600               # ms, hareketli noktaların gösterildiği süre
    mt_latency: int = 50            # ms, MT'nin harekete tepki gecikmesi (yaklaşık; MT gecikmesi onlarca ms)
    n_mt: int = 64                  # MT girdi nöronu sayısı (tercih yönleri 360 dereceye eşit dağılır)
    mt_spontaneous: float = 8.0     # Hz, fiksasyonda (hareket yokken) MT'nin kendiliğinden ateşleme hızı
    mt_baseline: float = 20.0       # Hz, %0 tutarlılıkta bile hareket MT'yi uyarır (Britten 1993)
    mt_gain: float = 30.0           # Hz / birim tutarlılık; MT hızı tutarlılıkla doğrusal artar (Britten 1993)
    frame_ms: int = 16              # ms, ekran karesi süresi (~60 Hz monitör)
    stim_noise: float = 0.5         # kare başına tutarlılık dalgalanması; rastgele noktaların doğası (model seçimi)
    decision_window: int = 200      # ms, karar için sayılan son pencere (model seçimi)
    # Laboratuvarda kullanılan standart tutarlılık seviyeleri (%0 - %51.2)
    coherences: tuple = (0.0, 0.032, 0.064, 0.128, 0.256, 0.512)

    @property
    def n_steps(self) -> int:
        return int((self.t_fix + self.t_stim) / self.dt)

    @property
    def stim_onset_step(self) -> int:
        return int(self.t_fix / self.dt)


@dataclass
class NetworkParams:
    """LIP bölgesi: uyarıcı (E) ve baskılayıcı (I) adaptif LIF nöronları."""

    n_exc: int = 160                # E nöronları; korteksin ~%80'i uyarıcıdır
    n_inh: int = 40                 # I nöronları; ~%20 baskılayıcı (80/20 oranı)
    pool_size: int = 40             # her seçim havuzu (Sol, Sağ) için E nöronu sayısı
    tau_mem_exc: float = 20.0       # ms, piramidal hücre membran zaman sabiti (tipik 10-30 ms)
    tau_mem_inh: float = 10.0       # ms, hızlı ateşleyen (FS) internöron; daha hızlı membran
    tau_syn: float = 5.0            # ms, hızlı sinapslar (AMPA, GABA_A) akım sönümü
    tau_nmda: float = 100.0         # ms, E -> E yavaş NMDA akımı; kanıt biriktirmeyi sağlar (Wang 2002)
    nmda_kappa: float = 0.5         # spike başına açılan NMDA reseptör oranı; g <- g + kappa (1 - g).
                                    # g <= 1 olduğundan akım doyuma ulaşır ve hızlar sınırsız tırmanmaz
    tau_adapt: float = 200.0        # ms, spike frekansı adaptasyonu (piramidal hücrelerde gözlenir)
    beta_adapt: float = 0.03        # her spike eşiği ne kadar yükseltir (model seçimi; büyük değer rampayı öldürür)
    v_th: float = 1.0               # normalize eşik
    refractory_ms: float = 2.0      # ms, mutlak refrakter dönem
    p_in: float = 0.5               # MT -> LIP bağlantı olasılığı (model seçimi)
    p_rec: float = 0.2              # LIP içi bağlantı olasılığı (lokal korteks ~%10-20)
    w_in: float = 0.08              # MT -> LIP başlangıç ağırlık ortalaması
    w_ee: float = 0.06              # E -> E
    w_plus: float = 48.0            # havuz içi E -> E güçlendirme; kazanan-hepsini-alır (Wang 2002 tarzı yapı)
    w_ei: float = 0.25              # E -> I
    w_ie: float = 0.50              # I -> E; ortak baskılama havuzları yarıştırır
    w_ii: float = 0.10              # I -> I
    bg_mean: float = 0.50           # arka plan akımı (korteksin geri kalanından gelen girdi)
    bg_std: float = 1.6             # arka plan gürültüsü -> düzensiz, Poisson benzeri ateşleme
    surrogate_slope: float = 10.0   # surrogate gradient eğimi (yalnızca ML eğitiminde önemli)

    @property
    def n_total(self) -> int:
        return self.n_exc + self.n_inh


@dataclass
class TrainParams:
    """Eğitim ayarları (iki yöntem için ortak + yönteme özel)."""

    method: str = "ml"              # "ml" (surrogate gradient) veya "bio" (R-STDP)
    iters: int = 300                # eğitim adımı sayısı
    batch_size: int = 32            # bir adımda paralel simüle edilen deneme sayısı
    seed: int = 42
    log_every: int = 10
    # --- ML (surrogate gradient) ---
    lr: float = 5e-3
    grad_clip: float = 1.0
    logit_temperature: float = 5.0  # Hz; havuz hızlarını logit'e çevirirken bölünen değer
    rate_max: float = 50.0          # Hz, üstü cezalandırılır (kortikal hızlar genelde < 50 Hz)
    rate_min: float = 1.0           # Hz, altı cezalandırılır (sessiz nöron istemiyoruz)
    rate_reg: float = 1e-3          # hız düzenlileştirme katsayısı
    # --- Biyolojik (R-STDP) ---
    eta: float = 0.15               # öğrenme hızı
    a_plus: float = 1.0             # STDP güçlenme genliği (LTP)
    a_minus: float = 0.5            # STDP zayıflama genliği (LTD); LTP baskın pencere
    tau_plus: float = 20.0          # ms, STDP zaman penceresi (önce -> sonra)
    tau_minus: float = 20.0         # ms, STDP zaman penceresi (sonra -> önce)
    tau_elig: float = 800.0         # ms, uygunluk izi süresi (dopamin gecikmesini köprüler)
    reward_avg_rate: float = 0.05   # ödül beklentisinin güncellenme hızı (eleştirmen)
    activity_avg_rate: float = 0.05 # nöron başına ortalama aktivitenin (kayan eşik) güncellenme hızı
    # --- Değerlendirme ---
    eval_trials: int = 100          # her (tutarlılık, yön) koşulu için deneme sayısı


@dataclass
class Config:
    task: TaskParams = field(default_factory=TaskParams)
    net: NetworkParams = field(default_factory=NetworkParams)
    train: TrainParams = field(default_factory=TrainParams)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        task = dict(d["task"])
        task["coherences"] = tuple(task["coherences"])
        return cls(
            task=TaskParams(**task),
            net=NetworkParams(**d["net"]),
            train=TrainParams(**d["train"]),
        )
