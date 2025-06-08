# Evaluacija Metodologij za Diplomsko Nalogo

Ta projekt je narejen v **Pythonu** in je namenjen evalvaciji različnih metodologij v okviru diplomske naloge. Osredotoča se na obdelavo podatkov, ki se nalagajo s pomočjo modula `src/load_quant.py`, in implementacijo prilagodljivega ogrodja (`src/framework.py`) za testiranje in primerjavo.

## Evalvacijski Pristopi

Projekt vključuje več skript za evalvacijo, ki pokrivajo naslednje pristope:
*   Osnovni: `src/evaluation_base.py`, `src/evaluation_base_half.py`
*   Dinamični: `src/evaluation_dynamic.py`
*   Statični: `src/evaluation_static.py`

Rezultati posameznih evalvacij so shranjeni v `.txt` datotekah v mapi `src/`.

## Pridobivanje modelov

V `src/main.py` se nahaja skripta, ki nauči modele uporabljene v diplomski nalogi in jih shrani.
Zahteve projekta so specificirane v `src/requirementsDiploma.txt`.