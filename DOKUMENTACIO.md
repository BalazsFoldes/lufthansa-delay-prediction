# Aviation Delay Prediction — Dokumentáció

**Feladat:** A járat késni fog-e több mint 15 percet?  
**Adat:** 6 000 soros, mesterségesen generált flight dataset  
**Nyelv:** Python | **Reprodukálhatóság:** `python run_pipeline.py`

---

## Tartalom

1. [Gondolkodásmód és megközelítés](#1-gondolkodásmód-és-megközelítés)
2. [EDA — Az adatok vizsgálata](#2-eda--az-adatok-vizsgálata)
3. [Modellezés](#3-modellezés)
4. [Kiértékelés — Baseline vs. fejlettebb modellek](#4-kiértékelés--baseline-vs-fejlettebb-modellek)
5. [Business Interpretation](#5-business-interpretation)

---

## 1. Gondolkodásmód és megközelítés

A feladatot a következő sorrendben közelítettem meg:

**Először az adatot vizsgáltam, nem a modellt.** Mielőtt bármilyen modellt tanítottam volna, végigmentem az összes oszlopon és megkérdeztem: *"Ez az információ elérhető lenne egy valódi indulás előtt?"* A data leakage vizsgálat lehet a legfontosabb egy projektben, és mint kiderült valóban nélkülözhetetlen volt a feladat során.

**Másodszor a metrikát választottam meg.** A célváltozó eloszlásának megnézése után azonnal nyilvánvaló lett, hogy az accuracy teljesen félrevezető lenne ennél a feladatnál. Ezt kellett először tisztázni, mielőtt bármit modellezek.

**Harmadszor a validációs stratégiát határoztam meg.** Mivel időbélyeges adatról van szó, random split helyett időalapú splitet alkalmaztam, ez szimulálja a valódi production körülményeket.

**Negyedszer jöttek a modellek**, de csak azután, hogy az előző három lépés tiszta volt.

Ez a sorrend szándékos: egy rossz adatminőségű vagy szivárgó pipeline-on tanított kiváló modell production-ban értéktelen.

---

## 2. EDA — Az adatok vizsgálata

### 2.1 Első áttekintés

Az adathalmazban 6 000 járat szerepel 27 oszloppal, 2024. január 1. és május 4. között. Az oszlopok három csoportba sorolhatók: repülési jellemzők, időjárási adatok és operációs mutatók.

### 2.2 Data Leakage — A legkritikusabb lépés

**Ez volt az első és legfontosabb vizsgálat.** Data leakage akkor áll fenn, ha a modell olyan információt lát tanítás közben, ami valódi deployment-ben nem lenne elérhető. Ez mesterségesen magas pontosságot produkál, ami production-ban azonnal összeomlik.

Az alábbi oszlopokat azonosítottam leakage-ként és távolítottam el **minden más lépés előtt:**

| Oszlop | Korreláció a targettel | Miért leakage |
|---|---|---|
| `actual_delay_minutes` | **+0.747** | A célváltozó ebből van számolva — tautológia |
| `actual_gate_out_time_diff` | **+0.710** | Csak az indulás után mérhető |
| `maintenance_closed_after_pushback` | +0.316 | Pushback utáni esemény |
| `final_delay_reason` | — | Késés után kerül hozzárendelésre; ahol NONE = nem késett, tehát tökéletes leakage |
| `sched_buffer_mins_latest` | — | 100%-ban azonos a `turnaround_minutes` oszloppal — redundáns |

**Megjegyzés az `ops_delay_prediction_v2`-ről:** A neve gyanúsnak tűnik, de a targettel mért korrelációja mindössze 0.12. Ez arra utal, hogy valódi előrejelzési metrika, nem post-hoc adat, ezért **bennhagytam** és a modell egyik bemenő jellemzőjeként kezelem.

### 2.3 Célváltozó eloszlása — Class Imbalance

Az adathalmazban súlyos osztályegyensúly-hiány áll fenn:

- **96% — nem késett** (5 762 járat)
- **4% — késett >15 percet** (238 járat)

Ez kritikus felismerés: egy modell, amely **mindig azt jósolja hogy nem lesz késés**, elér 96%-os accuracy-t, miközben egyetlen késést sem jelez előre. Ezért az accuracy teljesen alkalmatlan metrika ennél a feladatnál. Helyette **F1-score** és **ROC-AUC** metrikákat használok.

### 2.4 Hiányzó értékek

Két oszlopban találtam hiányzó értékeket:

| Oszlop | Hiányzó arány | Kezelés |
|---|---|---|
| `maintenance_events_last_30d` | 12.0% | Mediánnal pótolva |
| `visibility` | 7.7% | Mediánnal pótolva |

Mindkét esetben mediánt alkalmazok, mert ezek az értékek jobbra ferdített eloszlásúak és kiugró értékeket tartalmaznak, a medián robusztusabb becslést ad, mint az átlag. Fontos: az imputer **csak a training adaton lett fittelve**, majd alkalmazva a validációs és teszt setre.

### 2.5 Időbeli lefedettség

Az adatok 2024 januártól május elejéig fednek le 5 hónapot. Május csak részleges (192 járat), ami megerősíti az időalapú split szükségességét.

### 2.6 Mintázatok és fontos feature-ök

Az EDA során a következő jellemzőket találtam potenciálisan informatívnak:
- **`previous_leg_delay`** — az előző járat késése erősen korrelál a következőével
- **`airport_congestion_index`** — zsúfolt repülőtér késéseket propagál
- **`turnaround_minutes`** — kevés pufferidő esetén nincs helyreállási lehetőség
- **`crew_rest_hours`** — kifáradt személyzet operációs kockázatot jelent
- Időjárási változók (`visibility`, `wind_speed`, `precipitation`) — korlátozott befolyásolhatóság

---

## 3. Modellezés

### 3.1 Validációs stratégia — Időalapú split

Mivel a dataset időbélyeggel rendelkezik, **random split helyett időalapú splitet** alkalmazok:

```
Idővonal:  Jan ──── Feb ──── Mar ──── Apr ──── May
            [     TRAIN (4 368)   ] [ VAL ] [TEST]
```

| Szett | Időszak | Méret | Késési arány |
|---|---|---|---|
| Train | Jan–Mar 2024 | 4 368 járat | ~4% |
| Validáció | Ápr 2024 | 1 440 járat | ~4% |
| Test | Máj 2024 | 192 járat | ~4% |

**Miért nem random split?** Random splitnél a modell láthatna májusi adatot a tanítás során, miközben márciusi járatokat jósolna, ez valóságban lehetetlen. Az időalapú split szimulálja azt, ahogyan a modell élesben futna: mindig a múltból tanul és a jövőt jósolja.

### 3.2 Preprocessing

Minden transzformáció **kizárólag a training adaton lett fittelve**, majd alkalmazva a többi setre, ezzel megakadályozva a preprocessing-szintű adatszivárgást.

1. **Időbeli feature-ök** kinyerése a `flight_datetime`-ból:
   - hónap, hét napja, hétvége jelző
   - ciklikus órakódolás (sin/cos), hogy az éjfél és 23:00 szomszédosak legyenek
2. **Medián imputation** a hiányzó értékekre (train-en fittelve)
3. **One-hot encoding** kategorikus oszlopokra: `origin`, `destination`, `aircraft_type`
4. **StandardScaler** — csak a Logistic Regression-höz szükséges, a fa alapú modellek nem igénylik

### 3.3 Küszöbérték-hangolás
 
Mivel az osztályarány 96/4, az alapértelmezett 0.5-ös döntési küszöb extrém konzervatív, a modellek soha nem prediálják a pozitív osztályt. Ezért minden modellnél **automatikus küszöbkeresést** alkalmazok: 0.05 és 0.50 között keresem azt a küszöböt, ami a legjobb F1-score-t adja a validációs szetten. Ez a küszöb kerül ezután alkalmazásra a teszt szetten anélkül, hogy azt bármilyen hangolási döntésbe bevonnánk, szimulálva azt a production környezetet, ahol a döntési határt az új adatok látása előtt rögzítik. Ez a lépés nélkülözhetetlen imbalanced class esetén.

### 3.4 Modellek és indoklásuk

Hat modellt tanítottam két kategóriában:

#### Baseline modellek

**DummyClassifier (majority)**: Az igazi alap. Mindig a többségi osztályt (nem késik) jósolja. 96%-os accuracy-t ér el anélkül, hogy bármit tanulna. Bármely valódi modellnek ezt kell meghaladnia, és ez pontosan megmutatja, miért értéktelen az accuracy mint metrika.

**Logistic Regression**: Lineáris, interpretálható modell. `class_weight='balanced'` paraméterrel kezeli az osztályegyensúly-hiányt. Jó referenciapont arra, hogy mennyi lineárisan szeparálható jel van az adatban.

#### Fejlettebb modellek

**Decision Tree**: Nem-lineáris, emberi szemmel olvasható döntési szabályok. `max_depth=6` korlátozással a túlilleszkedés ellen. Hasznos annak vizsgálatára, hogy egyszerű if-then szabályok képesek-e elkülöníteni a késett járatokat.

**Random Forest**: 300 fa együttese. Robusztus, zajra kevésbé érzékeny, megbízható feature importance értékeket ad. `class_weight='balanced'` az imbalance kezelésére.

**XGBoost**: Gradiens boosting, általában a legjobb teljesítményt nyújtja tabuláris adaton. `scale_pos_weight=24` paraméter a 96/4 arány kompenzálására (5762/238 ≈ 24).

**LightGBM**: Az XGBoost-hoz hasonló, de natívan kezeli a hiányzó értékeket (releváns, mivel a `visibility` és `maintenance_events_last_30d` oszlopokban hiányok vannak), és gyorsabb a tanítása. `is_unbalance=True` flag az imbalance kezelésére.

---

## 4. Kiértékelés — Baseline vs. fejlettebb modellek
 
### 4.1 Eredménytábla (teszt szett — küszöbhangolással)

A küszöbértéket minden modellnél a **validációs szetten** kerestük meg,
és azt alkalmaztuk a teszt szetten anélkül, hogy a teszt adatot bármilyen
hangolási döntésbe bevontuk volna. Ez szimulálja azt a production környezetet,
ahol a döntési határt az új adatok látása előtt rögzítik.

| Modell | Accuracy | Precision | Recall | **F1** | **ROC-AUC** | Küszöb (val) |
|---|---|---|---|---|---|---|
| Dummy (majority) | 0.958 | 0.000 | 0.000 | 0.000 | 0.500 | 0.50 |
| **Logistic Regression** | **0.750** | **0.115** | **0.750** | **0.200** | **0.748** | **0.35** |
| Decision Tree | 0.620 | 0.067 | 0.625 | 0.121 | 0.565 | 0.25 |
| Random Forest | 0.896 | 0.000 | 0.000 | 0.000 | 0.710 | 0.40 |
| XGBoost | 0.875 | 0.056 | 0.125 | 0.077 | 0.654 | 0.20 |
| LightGBM | 0.802 | 0.031 | 0.125 | 0.050 | 0.673 | 0.15 |

*Elsődleges metrikák: F1-score és ROC-AUC. Az accuracy félrevezető a 96/4 osztályeloszlás miatt.*

### 4.2 Eredmények értelmezése

**A Logistic Regression a legjobb F1-en (0.200) és ROC-AUC-on (0.748).**
A `class_weight='balanced'` paraméter a kisebbségi osztály felé tolja a
becsült valószínűségeket, és a validációs szetten talált 0.35-ös küszöb
a teszt szetten is aktiválódik. Ez a stabilitás, val F1=0.161,
teszt F1=0.200, az egész összehasonlítás legfontosabb megfigyelése:
a modell valódi, általánosítható jelet talált, nem zajt.

**A Random Forest intő eredmény.** A validációs szetten a legjobb küszöb
0.40 volt (val F1=0.158). Ezt alkalmazva a teszt szetten a modell egyetlen
pozitív predikciót sem ad F1=0.000. A 0.710-es ROC-AUC ugyanakkor azt
mutatja, hogy a modell helyesen *rangsorolja* a késett járatokat, csak a
döntési határ nem általánosítható. Ez a szintetikus, véletlenszerű adat
közvetlen következménye: nincs stabil jel, amit az ensemble mindkét szetten
meg tudna fogni.

**Az XGBoost és LightGBM** alacsony küszöbökön (0.20, 0.15) aktiválódnak
ugyan, de F1-jük a teszt szetten minimális (0.077 és 0.050). A val szetten
mért F1 (0.167 és 0.146) sem volt erős, a teszt szetten pedig tovább romlott.

**Az accuracy mint metrika csapdája:** A Dummy modell 95.8%-os accuracy-t
ér el, magasabbat, mint a Logistic Regression (75.0%). Mégis teljesen
használhatatlan, mert egyetlen késést sem jelez előre. Ez szemléletes
bizonyítéka annak, miért kell iparágspecifikus metrikát választani.

**Fontos kontextus:** A teszt szetten mindössze 8 pozitív eset volt
(192 járatból). Emiatt a metrikák varianciája magas, 1-2 predikció
különbség is jelentősen változtatja az F1-t. A ROC-AUC
ebben az esetben stabilabb és megbízhatóbb összehasonlítási alap.

### 4.3 Melyik modell a legjobb?

A ROC-AUC és az F1 alapján egyaránt a **Logistic Regression** nyújtja
a legjobb és legstabilabb teljesítményt. A komplexebb ensemble modellek
nem tudtak a szintetikus adatban általánosítható jelet találni, ami
megerősíti, hogy a feladatban az adatelemzési folyamat és a módszertani
döntések értékesebbek a modell komplexitásánál.

---

## 5. Business Interpretation

### Mi okozza leginkább a késéseket?

A feature importance elemzés alapján (Random Forest és XGBoost):

1. **`previous_leg_delay`** — A legerősebb jel. Egy késett érkező járat kaszkádszerűen továbbterjeszti a késést. Ez az úgynevezett "rotation delay" — az egyik legismertebb jelenség a légiiparban.
2. **`airport_congestion_index`** — Zsúfolt repülőtéren a késések propagálnak: kevesebb szabad futópálya, torlódó gurulóutak, lassabb gate-kiosztás.
3. **`turnaround_minutes`** — Rövid fordulási idő esetén nincs puffer a helyreálláshoz. Ha az előző járat akár csak 10 percet késik, ez azonnal átterjed a következőre.
4. **`crew_rest_hours`** — Kevés pihenőidő operációs kockázatot jelent, és korrelál a késésekkel.
5. **Időjárási változók** (`visibility`, `wind_speed`, `precipitation`) — Valós hatással bírnak, de operációs oldalról nem befolyásolhatók.

### Mely tényezők befolyásolhatók operációs oldalról?

| Befolyásolható ✅ | Nem befolyásolható ❌ |
|---|---|
| **Fordulási idő (buffer) ütemezése**<br>`(turnaround_minutes)` | **Időjárási viszonyok**<br>`(visibility, wind_speed, precipitation, thunderstorm_flag)` |
| **Személyzet pihenőidejének menedzselése**<br>`(crew_rest_hours)` | **Útvonal távolsága**<br>`(route_distance)` |
| **Gate allokáció optimalizálása**<br>`(gate_availability)` | **Repülőtéri zsúfoltság és infrastruktúra**<br>`(airport_congestion_index, runway_utilization)` |
| **Karbantartás előzetes ütemezése**<br>`(maintenance_events_last_30d)` | **Légiforgalmi irányítás (ATC)**<br>*(Közvetlen adat nincs, de a hálózati terheltséggel összefügg)* |

A legértékesebb operációs beavatkozási pont a **previous_leg_delay figyelése**: ha az érkező járat már késik, az Ops Controller proaktívan átütemezhet gate-et, előkészítheti a gyors fordulást, vagy értesítheti a személyzetet.

### Mennyire bíznál a modellben?

**Őszintén ebben a formában kevésbé, de megfelelő okokból.**

Az adat szintetikusan generált, szándékos véletlenszerűséggel. Egy 0.748-as ROC-AUC véletlenszerű adaton inkább azt jelzi, hogy a modell megtalálta a kis mennyiségű beépített jelet, nem azt, hogy erős prediktív erővel bír.

Valódi operációs adaton, ahol tényleges ok-okozati összefüggések vannak (pl. valódi időjárási adatok, tényleges rotációs minták, historikus késési láncok), ugyanez a pipeline lényegesen jobb eredményt adna.

### Milyen limitációi vannak?

1. **Szintetikus adat** — Nincs valódi kauzális struktúra; az eredmények nem általánosíthatók valós adatra
2. **Kis teszt szett** — Csak 192 járat májusban, ebből ~8 késett; a metrikák magas varianciával bírnak
3. **Extrém class imbalance** — 96/4 arány megköveteli a küszöbhangolást production-ban
4. **Hiányzó külső adatok** — Nincs NOTAM, slot restriction, hálózati hatás, történelmi késési lánc
5. **Statikus modell** — A késési minták szezonálisan változnak; rendszeres retraining szükséges

### Hogyan használnád production környezetben?

**Mikor fusson a modell?**
T-60 perccel az ütemezett indulás előtt, ekkor már minden feature elérhető (időjárás, személyzet státusz, érkező járat késése), de még van idő beavatkozni.

**Döntési küszöb:**
Az alapértelmezett 0.5 helyett **0.15–0.20 körüli küszöb** ajánlott. A légiiparban egy kihagyott valódi késés (false negative) operációs és anyagi költsége lényegesen magasabb, mint egy felesleges riasztás (false positive). Magasabb recall-t érdemes vállalni alacsonyabb precision árán.

**Hogyan épüljön be a folyamatba?**
A modell kimenetét **jelzésként, nem automatikus döntésként** kezelem. A magas kockázatú járatokat egy Ops Controller elé tárom, aki a végső döntést meghozza. Ez a "human-in-the-loop" megközelítés különösen fontos, amíg a modell teljesítménye és megbízhatósága bizonyított.

**Monitoring és karbantartás:**
- Havonta: data drift figyelése (változnak-e a feature eloszlások?)
- Negyedévente: modell retraining gördülő 12 hónapos ablakkal
- Folyamatosan: precision/recall monitorozása éles predikciókra visszanézve

---

*A teljes technikai részletezés és vizualizációk az `analysis.ipynb` notebookban találhatók.*
*Az összes eredmény reprodukálható: `python run_pipeline.py`*