# Raport — parametry metody dostępne do drugiej ablacji

Data: 2026-08-10. Raport po polsku, bo jest dla Ciebie; ścieżki, nazwy pól i identyfikatory
zostają po angielsku. Dokumenty przekazywane drugiej instancji (`PARAMETER-STUDY-BRIEF.md`,
`ABLATION-STUDY-HANDOFF.md`) pozostają po angielsku.

Dwa słowa, które w tym dokumencie znaczą co innego niż mogłyby: **wzorzec startowy** to
gotowy układ toru wstawiany do populacji na starcie, a **seed** to liczba startowa
losowania, która odróżnia powtórzenia tego samego eksperymentu.

---

## 1. Skrót

Zbadałem, które parametry metody da się w tym projekcie realnie zmieniać, i podzieliłem je
według tego, ile kosztuje udostępnienie każdego z nich. Trzy ustalenia zmieniają sposób
czytania wyników:

1. `mutation_prob` nie oznacza „jak duża część chromosomu mutuje". Oznacza szansę, że
   osobnik dostanie **dokładnie jedną** drobną poprawkę.
2. W każdym z 21 configów `crossover_prob + mutation_prob = 1.0`, choć kod tego nie wymaga.
   Te dwie liczby działają jak jeden suwak, a jego położenie waha się między configami od
   0.2 do 0.95 — bez pomiaru, który by to uzasadniał.
3. Zestaw siedmiu operatorów mutacji działa wyłącznie dla układów bez zwrotnic i bez
   podwójnego rozjazdu. Dla pozostałych configów nie uruchamia się nigdy.

Kolejne czynniki warte zbadania, w tej kolejności: **suwak krzyżowanie/mutacja**, potem
**udział wzorców startowych w populacji**, potem **stopień wypełnienia losowych osobników**.

---

## 2. Jak to sprawdzałem

Wyłącznie czytając kod — repozytorium plus zainstalowany pakiet pymoo 0.6.1.6 w `.venv`.
Bez uruchamiania czegokolwiek, bo kampania ablacyjna trwa, a zmierzony czas jednego biegu
jest jedną z raportowanych przez nią wielkości. Nie edytowałem niczego w `src/`.

Każde twierdzenie o pymoo pochodzi z zainstalowanego źródła, nie z pamięci.

---

## 3. Trzy ustalenia, które zmieniają interpretację wyników

### 3.1 Mutacja jest znacznie słabsza, niż sugeruje liczba w configu

`src/operators.py:674` — `prob` przekazywane do pymoo działa **na osobnika**
`src/operators.py:692` — pętla po osobnikach; każdy dostaje co najwyżej jedną operację

`mutation_prob: 0.8` znaczy „80% osobników dostanie jedną drobną zmianę" — na przykład
podmianę typu jednego kawałka toru albo obrót jednego łuku. Nie znaczy „mutuje 80% genów".
Jak na algorytm genetyczny to bardzo delikatna mutacja i prawdopodobnie dlatego jedne
configi podkręcają ją aż do 0.8, a inne zjeżdżają do 0.05.

Jeżeli ten suwak wejdzie do siatki, wynik trzeba opisywać jako odsetek osobników
dostających jedną poprawkę, a nie jako siłę mutacji.

### 3.2 Krzyżowanie i mutacja są ustawiane jak jeden suwak

Sprawdzone we wszystkich 21 configach:

| wartości | configi |
|---|---|
| 0.2 / 0.8 | `default`, `with_switches`, `with_double_crossover_narrow`, `with_double_crossover_small` |
| 0.8 / 0.2 | `all_pieces_rect`, `switch_cross_rect`, `switch_one_siding_wide`, `switch_two_sidings_tall`, `with_switches_and_crossing` |
| 0.9 / 0.1 | `all_pieces`, `all_pieces_350x350`, `compact`, `cross_dc_rect`, `dc_figure8_large`, `dc_figure8_wide`, `plain_wide_racetrack`, `with_double_crossover` |
| 0.95 / 0.05 | `all_pieces_150x150`, `cross_figure8_tall`, `cross_figure8_wide`, `with_crossing` |

pymoo traktuje te dwa prawdopodobieństwa niezależnie i nic nie wymaga, żeby sumowały się
do jedynki. Ktoś przyjął taką konwencję, a rozrzut od „prawie wyłącznie krzyżujemy" do
„prawie wyłącznie mutujemy" nie ma za sobą żadnego pomiaru. To czyni z tej pary
najmocniejszego kandydata na kolejny czynnik: jest już w configu, więc nie wymaga zmian w
kodzie, a sama rozbieżność między configami domaga się wyjaśnienia.

### 3.3 Zestaw operatorów mutacji nie uruchamia się dla configów ze zwrotnicami

Mutacja rozgałęzia się na trzy przypadki:

`src/operators.py:698` — układ z aktywnym podwójnym rozjazdem: w 50% powiększenie
zachowujące domknięcie pętli, w 25% przebudowa ósemki, w 25% bez zmian
`src/operators.py:709` — układ z aktywną zwrotnicą: w 10% operacja na zwrotnicy, w
pozostałych 90% zawsze to samo powiększenie zachowujące domknięcie
`src/operators.py:717` — pozostałe układy: w 10% operacja na zwrotnicy, reszta losowana z
siedmiu operatorów pętli głównej według wag

Siedem wag zapisanych w `_MAIN_LOOP_WEIGHTS` (`src/operators.py:634`) dotyczy więc tylko
trzeciego przypadku. Dla `with_switches`, `dc_figure8_*` czy `switch_*` strojenie tych wag
nie zmieniłoby niczego, bo tam jedyną liczbą sterującą mutacją jest wspomniane 10%.

Zawężenie jest celowe i opisane w kodzie: pozostałe operatory rozrywają domknięcie pętli
wokół zwrotnicy, a zmiana opisu podwójnego rozjazdu praktycznie zawsze wyrzuca go z
układu. Nie jest to błąd do naprawienia, tylko fakt do uwzględnienia przy projektowaniu
badania.

---

## 4. Inwentarz parametrów

### 4.1 Dostępne od ręki — są w configu, zero zmian w kodzie

| parametr | co robi | uwaga |
|---|---|---|
| `pop_size` | ile układów algorytm trzyma naraz | już w siatce (§3.4 briefu) |
| `n_gen` | ile rund ulepszania wykonuje | już w siatce |
| `crossover_prob` / `mutation_prob` | suwak między krzyżowaniem a mutacją | patrz 3.1 i 3.2 |
| `heuristic_ratio` | jaka część startowej populacji to gotowe wzorce | twardy sufit 0.5 |
| `eliminate_duplicates` | czy usuwać powtórzone rozwiązania | dziś zawsze włączone |
| `termination.period` | czy przerywać bieg przy braku poprawy | kampanie wymuszają 0, żeby budżet był równy |

`src/config.py:93` — sufit `heuristic_ratio` to `le=0.5`; wyższa wartość kończy się błędem
walidacji

Miejsca, w których runner faktycznie czyta te pola:

`src/algorithm/runner.py:675` — `heuristic_ratio` trafia do generatora populacji startowej
`src/algorithm/runner.py:681` — `crossover_prob`
`src/algorithm/runner.py:682` — `mutation_prob`
`src/algorithm/runner.py:740` — `eliminate_duplicates`

### 4.2 Liczby wpisane na sztywno — wystarczy dopisać linię, żeby dało się je ustawiać

`src/sampling.py:706` — wypełnienie losowego osobnika, `rng.uniform(0.5, 0.8)`; losowy
układ zajmuje od 50% do 80% dostępnych miejsc w pętli. Nikt nie sprawdził, czy ten zakres
jest dobry.

`src/operators.py:690` — `junc_thresh = 0.10`; jaka część mutacji trafia w zwrotnice
zamiast w pętlę główną. Dla configów ze zwrotnicami jest to jedyna liczba sterująca
mutacją.

`src/operators.py:121` — 50% szansy, że dzieci wymienią się opisem podwójnego rozjazdu
`src/operators.py:129` — 50% szansy na wymianę opisu zwrotnicy
`src/operators.py:136` — 50% szansy na wymianę pozycji startowej

Te trzy piątki decydują o tym, jak mocno dzieci mieszają wyposażenie obojga rodziców.
Nigdy ich nie zmieniano.

`src/sampling.py:725` — w losowym osobniku każde miejsce na zwrotnicę włącza się rzutem
monetą
`src/sampling.py:728` — długość mijanki w losowym osobniku: od 0 do najwyżej 5 prostych

Liczba potomków na pokolenie równa się dziś wielkości populacji, bo tak zachowuje się
pymoo, gdy nie poda się `n_offsprings`:

`pymoo/algorithms/base/genetic.py:41` — `if self.n_offsprings is None: self.n_offsprings = pop_size`

Mniejsza wartość dałaby wariant, w którym populacja odnawia się stopniowo zamiast w
całości. Zmienia to jednak liczbę ocen w każdym pokoleniu, więc budżet przestaje być
prostym iloczynem wielkości populacji i liczby rund.

Sposób mierzenia zatłoczenia rozwiązań (`crowding_func`) jest w pymoo domyślnie ustawiony
na `cd`, choć dokumentacja samego pymoo poleca dla zadań dwucelowych `pcd`, a nasze zadanie
jest dwucelowe. **Sprawdzone i rozstrzygnięte: `pcd` nie jest lepsze.**

Porównanie poszło osobną osią ablacji, nie siatką parametrów — 10 wariantów × 21
konfiguracji × 3 ziarna × {cd, pcd}, czyli 1260 przebiegów. W ośmiu wariantach na dziesięć
`pcd` przegrywa więcej bloków, niż wygrywa, średnie różnice hipervolumenu mieszczą się w
przedziale −0.016..+0.003, żaden wariant nie osiąga `p < 0.5`, a czas liczenia jest ten sam
(19.7 h wobec 20.2 h). Domyślną wartością zostaje `cd`; szczegóły w
`ABLATION-STUDY-HANDOFF.md` §8.1.

`src/algorithm/runner.py` — `_build_survival` tworzy operator decydujący, kto przeżywa do
następnego pokolenia, i przyjmuje nazwę metryki z `AlgorithmConfig.crowding_func`

Naprawa układu ma dwa etapy, które pipeline potrafi włączać osobno, ale runner wpisuje oba
na sztywno:

`src/algorithm/runner.py:691` — `enable_closure_repair=True`
`src/algorithm/runner.py:692` — `enable_boundary_repair=True`

Ablacja składników włącza i wyłącza całą naprawę naraz. Rozbicie jej na dwa etapy kosztuje
dwie linie i odpowiada na pytanie, którego nikt dotąd nie zadał: który etap naprawy
naprawdę pracuje.

Harmonogram stopniowego zaostrzania wymagań wobec rozwiązań niedopuszczalnych opiera się na
pięciu liczbach, z których żadna nie jest w configu:

`src/algorithm/runner.py:756` — `hold_until=0.2`, `perc_eps_until=0.9`
`src/algorithm/runner.py:536` — `theta=0.2`, `ratchet_trigger=0.25`, `ratchet_cooldown=5`
`src/algorithm/runner.py:449` — `HARD_CONSTRAINT_WEIGHT = 1000.0`

### 4.3 Rzeczy, których nie da się dziś ustawić w żaden sposób

**Skład wzorców startowych, a nie ich liczba.** Wzorce ze wszystkich dziewięciu rodzin są
tasowane, a potem rozdzielane cyklicznie, czyli po równo:

`src/sampling.py:647` — `patterns[i % len(patterns)]`

Dla configu z podwójnym rozjazdem oznacza to, że większość miejsc zajmują zwykłe owale i
pętle w kształcie toru wyścigowego, które temu zadaniu nic nie dają. „Więcej wzorców
właściwej rodziny" to inne pytanie niż „więcej wzorców w ogóle" i wymaga nowego pola w
configu oraz zmiany w generatorze.

**Wagi siedmiu operatorów mutacji.** Siedem liczb sumujących się do jedynki nie nadaje się
na czynnik w siatce. Sensowniej jest wyłączać po jednym operatorze i patrzeć, co się psuje
— to mała ablacja operatorów, a nie strojenie jednej liczby.

### 4.4 Czego nie wolno tu ruszać

`src/problem.py:33` — `SPEED_SAFETY_MARGIN = 0.95`

Ta stała, razem z `special_piece_weight` oraz tolerancjami domknięcia i granicy, definiuje
**zadanie**, a nie metodę jego rozwiązywania. Zmiana którejkolwiek przesuwa wartości celów
i granicę dopuszczalności, więc wyniki przestają leżeć na wspólnej skali — a cała analiza
jakości na wspólnej skali się opiera. Inwentarz i obszar, w którym układ ma się zmieścić,
są warunkami zadania danymi z zewnątrz, nie pokrętłami.

Siły selekcji też nie da się ustawić, choć wygląda na parametr: funkcja turnieju w pymoo
jest napisana wyłącznie pod pojedynek dwóch osobników i przy innej liczbie kończy się
wyjątkiem.

`pymoo/algorithms/moo/nsga2.py:25` — `raise ValueError("Only implemented for binary tournament!")`

---

## 5. Kolejne czynniki warte zbadania

1. **Suwak krzyżowanie/mutacja.** Jest już w configu, rozrzut między configami ogromny, a
   dowodów na żadne ustawienie nie ma. Udostępnienie nic nie kosztuje.
2. **Udział wzorców startowych w populacji.** Ważny, bo ablacja składników pokazała, że to
   zasiew populacji startowej daje największy efekt. Liczba wzorców to jednak procent razy
   wielkość populacji, więc czynnik ten jest sprzężony z wielkością populacji i bez
   dodatkowych biegów nie da się ich rozdzielić.
3. **Wypełnienie losowych osobników, dziś od 50% do 80%.** Tanie w udostępnieniu, jedna
   linia, a nikt nie wie, czy ten zakres wynika z analizy, czy z przypadku.

Dla configów ze zwrotnicami i podwójnym rozjazdem zamiast pozycji trzeciej sensowniejszy
jest podział budżetu mutacji, czyli wspomniane 10%, bo pozostałe operatory i tak się tam
nie uruchamiają.

Wszystko poza pozycją pierwszą wymaga zmian w `src/`, a te są zamrożone do końca kampanii.
Projektujemy i ustawiamy w kolejce.

---

## 6. Co już zostało rozstrzygnięte

- Liczba rund (`n_gen`) jest badanym czynnikiem, a nie budżetem do ochrony. Zasada „nie
  skracaj biegów" dotyczy oszczędzania na kampanii, a nie sytuacji, w której liczba rund
  jest właśnie tym, co mierzymy.
- Siatka ma dwa pokrętła: wielkość populacji razy liczba rund, każde w połowie i w
  podwojeniu wartości produkcyjnej danego configu. Cztery rogi, trzy seedy, koszt 6.25
  zwykłego biegu na jeden config i jeden seed.
- Środka siatki nie trzeba liczyć — wariant `full` z kampanii ablacyjnej przejechał
  dokładnie ustawienia produkcyjne na wszystkich configach i trzech seedach.
- `pcd` przeciwko `cd` sprawdzamy osobnym wariantem w ablacji składników, nie w siatce.

Wszystko to jest już wpisane do `PARAMETER-STUDY-BRIEF.md`, sekcje 3.1 i 3.4.

---

## 7. Czy nasze operatory robią to, czego pymoo od nich wymaga

Sprawdzałem to na kodzie klas bazowych zainstalowanego pymoo 0.6.1.6, a nie na
dokumentacji — strony `pymoo.org/operators/*` pokazują przykłady użycia, ale nie mówią,
czego biblioteka od własnego operatora wymaga. Wszystkie trzy nasze operatory te wymagania
spełniają.

**Krzyżowanie.** Dostajemy i zwracamy tablicę w kształcie, jakiego pymoo oczekuje;
biblioteka to sprawdza i przerwałaby bieg, gdyby się nie zgadzał. Wartości `crossover_prob`
nasz kod w ogóle nie sprawdza, bo losuje ją sama biblioteka. Para rodziców, która nie
przeszła losowania, nie jest krzyżowana i do potomstwa trafiają ich niezmienione kopie.

`pymoo/core/crossover.py:51` — sprawdzenie kształtu tablicy z wynikiem
`pymoo/core/crossover.py:66` — kopiowanie rodziców dla par bez krzyżowania

**Mutacja.** Potwierdza się ustalenie z §3.1: pymoo losuje raz na osobnika, a nie raz na
gen. Nasz operator zmienia tablicę w miejscu, co bywa niebezpieczne, ale tutaj jest
poprawne, bo biblioteka podaje kopię danych, a nie oryginalną populację.

`pymoo/core/mutation.py:32` — losowanie raz na osobnika
`pymoo/core/population.py:72` — populacja jest kopiowana przy odczycie

**Naprawa.** Zgodna. Nasz łańcuch czterech napraw wywołuje wewnętrzne metody kolejnych
etapów zamiast ich publicznych odpowiedników i tak właśnie trzeba, bo metoda publiczna
oczekuje całej populacji, a łańcuch przekazuje dalej samą tablicę liczb.

`pymoo/core/repair.py:13` — biblioteka podaje tablicę i wstawia zwrócony wynik z powrotem

### 7.1 Przy niskim prawdopodobieństwie mutacji większość pracy idzie do kosza

pymoo najpierw każe zmutować **całą** populację, a dopiero potem losuje, których osobników
zmiana faktycznie dotyczy, i resztę odrzuca. Własnym operatorom biblioteki to nie
przeszkadza, bo przetwarzają całą tablicę naraz i jest to tanie. Nasze mutacje chodzą po
osobnikach pojedynczo, a niektóre z nich są kosztowne. Najdroższa prostuje tor w okolicy
miejsca, w którym tor przecina sam siebie — żeby takie miejsce znaleźć, musi najpierw
zbudować cały układ, a potem porównać każdy odcinek z każdym innym, więc jej koszt rośnie
z kwadratem liczby kawałków.

Przy `mutation_prob: 0.05`, czyli w `with_crossing` i obu configach `cross_figure8_*`, około
95% tej pracy zostaje wykonane i wyrzucone.

Da się to obejść: ustawić prawdopodobieństwo na 1.0 i przenieść losowanie do środka naszego
operatora. Wynik byłby statystycznie taki sam, a liczylibyśmy tylko tych osobników, których
zmiana naprawdę dotyczy. Zmieniłoby to jednak znaczenie liczby zapisanej w configach, więc
na czas badania parametrów tego nie ruszamy.

### 7.2 Naprawa działa też na populacji startowej

Tego nie ma w dokumentacji, a zmienia sposób czytania obu badań. pymoo używa operatora
naprawy w dwóch miejscach: przy tworzeniu potomstwa i przy budowaniu populacji startowej.

`pymoo/algorithms/base/genetic.py:56` — naprawa wpięta w budowanie populacji startowej
`pymoo/algorithms/base/genetic.py:64` — naprawa wpięta w tworzenie potomstwa
`pymoo/core/initialization.py:39` — naprawiana jest cała populacja startowa
`pymoo/core/initialization.py:42` — powtórzone osobniki są usuwane dopiero po naprawie

Trzy skutki:

1. Gotowe wzorce startowe są oceniane dopiero **po naprawie**. Zdanie „wzorzec startowy jest
   już najlepszym rozwiązaniem" znaczy więc naprawdę „naprawiony wzorzec jest najlepszy", a
   surowy wzorzec mógł się w ogóle nie domykać.
2. Zasiew i naprawa nie są w pierwszym pokoleniu niezależne. Wariant bez naprawy ocenia
   surowe wzorce, a wariant bez zasiewu ocenia naprawione układy losowe. To, co mierzymy
   jako wpływ zasiewu, jest wpływem zasiewu przepuszczonego przez naprawę.
3. Powtórzone osobniki są usuwane po naprawie i **nikt nie dosypuje nowych w ich miejsce**.
   Jeśli naprawa zamieni dwa różne układy w identyczne, pierwsze pokolenie będzie mniejsze
   niż zamówiona wielkość populacji, a rzeczywisty udział wzorców startowych może się różnić
   od wpisanego w `heuristic_ratio`. Dotyczy to wprost obu liczb, które kręcimy w siatce.

Skutku trzeciego nikt nie zmierzył. Do tego powstała komenda `/seed-repair-audit`.

---

## 8. Ograniczenia tego raportu

- To jest analiza kodu, a nie pomiar. Żadnej z tych liczb nie sprawdzono biegiem; raport
  mówi, co **da się** zmieniać i co dana wartość **znaczy**, a nie która wartość jest
  lepsza.
- Nie sprawdzałem, czy operatory sensownie znoszą wartości skrajne — na przykład czy
  `crossover_prob = 0` nie prowadzi do zastoju populacji. Skrajne poziomy trzeba będzie
  najpierw przepuścić przez zwykły bieg, zanim wejdą do siatki.
- Koszty podane w briefie pochodzą ze zmierzonych czasów istniejących biegów. Dla configów
  spoza tamtej tabeli pomiaru nie ma i nie wolno go zgadywać.

---

## 9. Stan kampanii w momencie pisania

Sprawdzone 2026-08-10 o 20:17: kampania seedów 2 i 3 nadal biegnie, katalogów z biegami
jest 200, a przed jej startem było 127 z seeda 1. Daje to około 73 wykonane biegi z 252 w
ciągu 85 minut, czyli tempo mniej więcej 0.86 biegu na minutę i zakończenie w okolicach
północy. To oszacowanie z tempa, nie deklaracja.

Warunek zakończenia pozostaje jeden: `outputs/ablation/manifest.json` musi zawierać wiersze
dla seedów 2 i 3. Plik jest zapisywany raz, na końcu.