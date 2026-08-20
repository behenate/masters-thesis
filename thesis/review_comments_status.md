# Rejestr uwag do `main_complete_1_komentarze.pdf`

Statusy: **naprawiono** - uwaga została uwzględniona w pracy; **częściowo** - poprawiono opis i ograniczenia, ale pełne rozwiązanie wymaga nowego eksperymentu lub danych; **do decyzji** - potrzebne jest wyjaśnienie autora komentarza.

Jeżeli komentarz zawierał gotowe brzmienie zdania lub terminu, wykorzystano je
bezpośrednio albo z niewielką korektą składni i formatowania LaTeX. Dotyczy to w
szczególności uwag 3--4, 10--14, 17, 19--20, 23--26, 28, 32, 37, 39--40, 43,
49--51, 54--55 oraz 61--62. Propozycji nie kopiowano dosłownie, gdy opisywała
eksperyment, którego nie wykonano, jak w uwagach 5, 41, 44, 48 i 52.

| Nr | Strona | Treść uwagi / wskazany fragment | Status | Sposób rozwiązania |
|---:|---:|---|---|---|
| 1 | 2 | Oświadczenie powinno być składane jako osobny dokument. | Naprawiono | Usunięto stronę oświadczenia z `main.tex` i pozostawiono komentarz wyjaśniający. |
| 2 | 5 | „w przeciwieństwie do wielu klasycznych” | Naprawiono | Zastąpiono nieprecyzyjne przeciwstawienie opisem reprezentacji zależnych od kontekstu. |
| 3 | 5 | „jaki jest ich koszt w porównaniu do klasycznych klasyfikatorów” | Naprawiono | Poprawiono składnię na „w porównaniu z klasycznymi klasyfikatorami”. |
| 4 | 6 | Zdanie o wykorzystaniu ustawień do porównania czterech metod. | Naprawiono | Wymieniono w jednym zdaniu wszystkie cztery metody w tej samej kolejności co w części badawczej. |
| 5 | 6 | Pytanie badawcze powinno odnosić się do końcowego testu i właściwych metryk. | Częściowo | Pytanie dostosowano do rzeczywistego protokołu: próbki treningowej i zewnętrznej walidacji; wspólny niezależny test końcowy nie został wykonany. |
| 6 | 8 | „Do usunięcia” - pusta strona. | Naprawiono | Ustawiono `open=any`, dzięki czemu rozdział nie wymusza pustej strony. |
| 7 | 9 | Brak odstępów przed odwołaniami bibliograficznymi. | Naprawiono | Ujednolicono zapis odstępów przed `\autocite` w plikach pracy. |
| 8 | 9 | „aby zasymulować prawidłowe logowanie” | Naprawiono | Przeredagowano zdanie tak, aby opisywało podszycie się pod stronę logowania, a nie symulowanie poprawnego logowania. |
| 9 | 9 | „elementy psychologiczne, takie jak presja czasu” | Naprawiono | Uproszczono sformułowanie i powiązano presję czasu bezpośrednio z nakłanianiem odbiorcy do działania. |
| 10 | 10 | Niejasny opis klas w klasyfikacji binarnej. | Naprawiono | Jednoznacznie określono klasę dodatnią jako spam, a ujemną jako wiadomość pożądaną. |
| 11 | 10 | Terminologia `false positive`. | Naprawiono | Wprowadzono „błąd fałszywie dodatni (ang. *false positive*, FP)”. |
| 12 | 10 | Terminologia `false negative`. | Naprawiono | Wprowadzono „błąd fałszywie ujemny (ang. *false negative*, FN)”. |
| 13 | 11 | Czas gramatyczny w opisie klasyfikacji tekstu. | Naprawiono | Zmieniono „zamieniano” na teraźniejsze „zamienia się”. |
| 14 | 11 | Rysunek BPE sugeruje tokenizer użyty w Qwen3. | Naprawiono | Tytuł i podpis wskazują, że jest to dydaktyczny przykład byte-level BPE z GPT-2; dopisano, że Qwen3 używa własnego tokenizatora. |
| 15 | 12 | „1?” przy numerze w listingu. | Naprawiono | Wyłączono numery linii w listingu. |
| 16 | 12 | „1?” przy kolejnym numerze w listingu. | Naprawiono | Wyłączono numery linii w listingu. |
| 17 | 12 | Nieprecyzyjne wyjaśnienie UTF-8. | Naprawiono | Wyjaśniono przejście od znaków Unicode do bajtów UTF-8 i rolę słownika 256 bajtów. |
| 18 | 12 | Brak objaśnienia tokenu `<unk>`. | Naprawiono | Dodano informację, że reprezentacja bajtowa pozwala uniknąć osobnego tokenu nieznanego. |
| 19 | 13 | Cyfra „4” w tekście ciągłym. | Naprawiono | Zastąpiono ją słowem „cztery”. |
| 20 | 15 | F1 nie uwzględnia TN i wymaga kontekstu pozostałych metryk. | Naprawiono | Dodano zastrzeżenie o wspólnej interpretacji F1, swoistości, FPR i rozkładu klas. |
| 21 | 16 | `think` i `tool_call` nie są rolami czatu. | Naprawiono | Opisano je jako bloki treści odpowiedzi, oddzielając od ról `system`, `user` i `assistant`. |
| 22 | 16 | Listing ChatML: rzeczywisty szablon, znaki specjalne i kodowanie. | Naprawiono | Listing oznaczono jako schemat zgodny z ChatML, poprawiono tokeny specjalne i usunięto problematyczne polskie znaki z przykładu. |
| 23 | 17 | Szyk „zwróć ... po ocenie”. | Naprawiono | Zmieniono na „oceń ... i zwróć”. |
| 24 | 17 | Szyk zdania o porównaniu wariantów klasyfikacji. | Naprawiono | Zmieniono na „przy porównaniu testowanych wariantów klasyfikacji”. |
| 25 | 20 | Głowy uwagi a analiza wykonana w pracy. | Naprawiono | Wyjaśniono możliwość uczenia różnych relacji oraz zaznaczono, że pracy nie obejmowała analiza pojedynczych głów. |
| 26 | 21 | Złożoność kwadratowa dotyczy macierzy uwagi. | Naprawiono | Ograniczono stwierdzenie do macierzy uwagi i dopisano zależność całkowitego kosztu od wymiaru ukrytego i implementacji. |
| 27 | 21 | Angielskie nazwy odmian attention. | Naprawiono | Dodano polskie odpowiedniki i pozostawiono angielskie terminy pomocniczo. |
| 28 | 22 | „Taki trening ma też słabszą stronę”. | Naprawiono | Zastąpiono sformułowaniem „Ograniczeniem takiego treningu jest ryzyko...”. |
| 29 | 26 | Szacunek pamięci dla Adam jest zbyt bezwarunkowy. | Naprawiono | Wskazano składniki zużycia pamięci oraz zależność od precyzji, wariantu optymalizatora i implementacji. |
| 30 | 28 | „Do usunięcia” - pusta strona. | Naprawiono | Ustawienie `open=any` usuwa wymuszoną pustą stronę między rozdziałami. |
| 31 | 29 | SOTA powinien uwzględniać aktualne prace o LLM w filtrowaniu spamu. | Naprawiono | Dodano dwa nowsze badania dotyczące klasyfikacji spamu przez modele językowe oraz określono zakres niepokryty przez te prace. |
| 32 | 30 | Literówka „stosuknowo”. | Naprawiono | Poprawiono na „stosunkowo”. |
| 33 | 30 | Zbędny przecinek. | Naprawiono | Poprawiono składnię zdania. |
| 34 | 31 | Zbyt ogólny opis fastText. | Naprawiono | Opisano n-gramy znakowe, reprezentację dokumentu, klasyfikator liniowy i dokładną konfigurację eksperymentu. |
| 35 | 35 | Podać dokładny identyfikator modelu; usunąć niezgodne „Base”. | Naprawiono | Podano `Qwen/Qwen3-0.6B` i usunięto niejednoznaczne określenie wariantu. |
| 36 | 35 | Tabela miesza warianty Base/IT i różne protokoły benchmarków. | Naprawiono | Usunięto liczbową tabelę rankingową oraz oparty na niej ranking; pozostawiono porównanie techniczne rodzin modeli. |
| 37 | 35 | Zbyt kategoryczny wniosek o pojemności najmniejszych modeli. | Naprawiono | Oznaczono go jako niezweryfikowaną hipotezę i nie użyto jako wyniku eksperymentalnego. |
| 38 | 35 | Kryterium mapowania reklam do klasy spam. | Naprawiono | Opisano mapowanie bez ręcznej zmiany etykiet oraz wskazano, że wykorzystano etykiety źródłowe. |
| 39 | 36 | Enron nie jest wiarygodnie zbiorem wyłącznie ham. | Naprawiono | Opisano to jako przybliżenie, dodano ręczny audyt 100 wiadomości i zastrzeżenie o możliwym szumie etykiet. |
| 40 | 37 | Charakter `fraudulent_email_corpus`. | Naprawiono | Nazwano go zbiorem wiadomości oszukańczych, w tym oszustw typu *advance-fee fraud*. |
| 41 | 40 | Brak końcowych metryk na nietkniętym zbiorze testowym dla wszystkich metod. | Częściowo | Podano liczebności i role splitów oraz jawnie wskazano brak wspólnej ewaluacji końcowej; jej wykonanie wymaga adapterów wszystkich metod i nowego pomiaru. |
| 42 | 41 | „Wiadomości prawdziwe/legalne”. | Naprawiono | Ujednolicono terminologię do „wiadomości pożądane” lub `ham`. |
| 43 | 41 | Znaczenie etykiety 0. | Naprawiono | Jednoznacznie zapisano, że `0` oznacza wiadomość pożądaną (`ham`). |
| 44 | 42 | Deduplikacja grupowa, konflikty etykiet i kolejność względem splitu. | Częściowo | Opisano rzeczywisty algorytm, wykryto siedem grup konfliktowych i podano sposób zachowania pierwszego rekordu; poprawne rozstrzygnięcie konfliktów wymagałoby przebudowy danych i ponownego treningu. |
| 45 | 42 | Brak parametrów i przykładów deduplikacji. | Naprawiono | Dodano tabelę parametrów, liczbę grup, największą grupę oraz przykładowe tematy z grup duplikatów. |
| 46 | 44 | Etykiety osi na wykresie liczebności duplikatów. | Naprawiono | Wykres używa całkowitych etykiet liczbowych i czytelnego zakresu osi. |
| 47 | 47 | Szczegóły metody następnego tokenu. | Naprawiono | Dodano wyłączenie thinking, rzeczywisty prompt, identyfikatory tokenów `ham`/`spam` i softmax ograniczony do dwóch etykiet. |
| 48 | 47 | Protokół modelu bazowego i surowe odpowiedzi. | Częściowo | Dodano parametry generacji, zachowanie parsera, ten sam szablon dla scoringu i ręczną kontrolę odpowiedzi; nie przeprowadzano osobnego wariantu few-shot ani analizy przykładowych logitów. |
| 49 | 49 | Niezręczne „seed'a”. | Naprawiono | Zmieniono na „ziarno losowe równe 67”. |
| 50 | 50 | `train_subset` nie mierzy generalizacji. | Naprawiono | Nazwano go próbką treningową służącą wyłącznie kontroli dopasowania. |
| 51 | 52 | Zbiory użyte przy wyborze modelu nie są końcowym testem. | Częściowo | W całej analizie nazwano je zewnętrzną walidacją i usunięto twierdzenia o niezależnym teście; niezależnego zbioru końcowego nie dodano. |
| 52 | 53 | Wybór punktu kontrolnego na zbiorach zewnętrznych narusza niezależność testu. | Częściowo | Jawnie opisano wybór na podstawie zewnętrznej walidacji i ograniczono wnioski; pełne rozwiązanie wymaga osobnego testu końcowego. |
| 53 | 53 | Brak czasu treningu konfiguracji K16. | Naprawiono | Dodano K16 do wykresu i zaktualizowano średni czas oraz przepustowość dla kontekstu 1024. |
| 54 | 59 | FPR 3,1% może być kosztowny przy dużym wolumenie. | Naprawiono | Przeliczono wynik na około 31 fałszywych alarmów na 1000 wiadomości i wskazano potrzebę doboru progu do kosztów wdrożenia. |
| 55 | 59 | Jeden wariant `r=8` nie uzasadnia ogólnego wniosku. | Naprawiono | Usunięto uogólnienie i zapisano potrzebę powtórzeń dla większej liczby konfiguracji i ziaren. |
| 56 | 61 | Numery metod w tabelach i wykresach są mało czytelne. | Naprawiono | Zastąpiono oznaczenia `01`--`04` nazwami sposobów klasyfikacji. |
| 57 | 66 | Niejasna relacja między 10\% walidacji a późniejszą wzmianką o 2\%. | Naprawiono | Ujednolicono opis do podziału 80/10/10, podano liczebności i role podzbiorów oraz wyjaśniono, że metody bazowe korzystały z całej 10-procentowej części walidacyjnej. |
| 58 | 67 | Brak Naive Bayes w tabeli konfiguracji. | Naprawiono | Dodano wiersz Naive Bayes z użytymi ustawieniami. |
| 59 | 69 | Niejasna definicja czasu ładowania. | Naprawiono | Zdefiniowano jego początek, koniec i wskazano, że nie jest doliczany do inferencji. |
| 60 | 72 | „Do usunięcia” - pusta strona. | Naprawiono | Ustawienie `open=any` usuwa pustą stronę przed kolejnym rozdziałem. |
| 61 | 73 | Wartość F1 bez symbolu procenta. | Naprawiono | Zmieniono zapis na `0\%`. |
| 62 | 74 | Porównanie kosztów różni się budżetem strojenia i rozmiarem partii. | Naprawiono | Dodano zastrzeżenie, że pomiar jest orientacyjny i nie stanowi pełnej analizy kosztu wdrożenia. |
| 63 | 74 | Ograniczenia powinny znajdować się przy właściwych metodach i wynikach. | Naprawiono | Najważniejsze zastrzeżenia umieszczono przy deduplikacji, walidacji, profilu błędów i kosztach, a następnie krótko zebrano w podsumowaniu. |
| 64 | 78 | Źródło formatu ChatML/Qwen. | Naprawiono | Dodano odwołanie do oficjalnej karty modelu Qwen3 i oznaczono listing jako schemat. |
| 65 | 80 | Pochodzenie wersji zbiorów z Kaggle. | Częściowo | Dodano nazwy plików, odwołania do pierwotnych źródeł i manifest SHA-256; numery wersji kopii Kaggle nie były zapisane podczas pobierania. |
| 66 | 82 | „Do usunięcia” przy angielskim streszczeniu. | Do decyzji | Nie usuwano streszczenia. Trzeba zapytać autora uwagi, czy chodzi o cały rozdział `Summary`, czy o konkretny element jego składu. |

## Pozostałe działania wymagające decyzji lub nowego eksperymentu

1. Zapytać autora komentarza 66, co dokładnie należy usunąć w części `Summary`.
2. Wykonać wspólną ewaluację wszystkich metod na nietkniętym zbiorze testowym, jeżeli ma ona zostać dodana do pracy (komentarze 5, 41, 51 i 53).
3. Zdecydować, czy przebudować zbiór po rozstrzygnięciu konfliktów etykiet w grupach duplikatów; taka zmiana wymaga ponownego treningu (komentarz 44).
4. Ustalić, czy potrzebne są dodatkowe warianty modelu bazowego, np. few-shot lub prezentacja przykładowych logitów (komentarz 48).
5. Odtworzyć numery wersji kopii Kaggle, jeżeli są dostępne w historii pobierania lub metadanych konta (komentarz 65).
