# Pytania do recenzenta

- [ ] Komentarz 66, s. 82: wyjaśnić, czy uwaga „Do usunięcia” dotyczy całej
  anglojęzycznej sekcji `Summary`, czy tylko konkretnego elementu na tej
  stronie. Do czasu uzyskania odpowiedzi nie usuwać `tex/summary.tex`.

# Pozostałe zadania badawcze

- [x] Ponownie przeprowadzić ewaluację modelu `Qwen/Qwen3-0.6B` bez
  dostrajania. Zastosowano doprecyzowany prompt `decision-checklist`, wyłączono
  tryb thinking i zapisano surowe odpowiedzi oraz reguły parsera dla czterech
  zbiorów po 1000 wiadomości. Parser odczytał wszystkie odpowiedzi. Model zaczął
  używać obu klas, ale nie osiągnął zakładanej jakości: accuracy wyniosło 58,3\%
  na `train_subset`, specificity 18,7\% na `enron`, recall 96,3\% na
  `fraudulent_email_corpus`, a accuracy 51,3\% na `spam_ham`. Wyniki znajdują
  się w `methods/00_unmodified_model/results/runs/decision_checklist_final_1000_seed67/`.
  Przed zmianą treści pracy należy zdecydować, czy ten wariant zastępuje
  dotychczasowy pomiar zero-shot, czy zostaje opisany jako dodatkowa analiza
  wrażliwości na prompt.
