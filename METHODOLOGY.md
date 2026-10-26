# Méthodologie — PEA RV Screener

Univers : ~275 actions dont le siège est dans l'UE/EEE, cotées en Europe (`universe.csv`). Exclus : Royaume-Uni,
Suisse, SIIC/REIT/SOCIMI (exonérées d'IS donc non éligibles PEA). Les lignes nordiques (SEK/DKK/NOK) sont
éligibles mais pas proposées par tous les courtiers en PEA : signalées `nordic_listing`.
Tous les calculs statistiques se font sur prix convertis en EUR ; les niveaux d'entrée/TP/SL sont en devise locale.

Le PEA interdit la vente à découvert : chaque signal RV se traduit par l'achat de la jambe bon marché seule.
On garde donc l'exposition marché/secteur, d'où l'overlay macro.

## 1. Paires cointégrées (PAIR)
- Présélection dans chaque secteur : corrélation des rendements journaliers sur 252 j ≥ 0,5 (esprit Gatev–Goetzmann–Rouwenhorst).
- Test d'Engle-Granger (`statsmodels.coint`, MacKinnon) sur log-prix 252 j, p < 0,05 ; ratio de couverture OLS.
- Spread modélisé en Ornstein-Uhlenbeck via AR(1) : moyenne d'équilibre m, σ_eq, demi-vie ∈ [3 ; 60] j.
- z = (spread − m) / σ_eq. Signal à |z| ≥ 2, watch à 1,5. On achète la jambe sous-évaluée.
- Robustesse : cointégration aussi sur 504 j, Hurst du spread, dérive du ratio de couverture (moitié 1 vs moitié 2),
  bêta Kalman final.
- Backtest walk-forward hors échantillon : paramètres estimés sur 252 j glissants, réestimés tous les 21 j ;
  entrée z ≤ −2, sortie z ≥ −0,5, stop z ≤ −3,5 ou 2× demi-vie. Rendement long seul et rendement couvert.

## 2. Résidu sectoriel (RESID) — Avellaneda & Lee (2010)
- Régression sur 60 j du rendement du titre sur le panier équipondéré de son secteur (hors titre).
- Résidu cumulé → AR(1) → OU ; filtre κ > 252/30 (demi-vie < ~30 j).
- s-score = (X − m) / σ_eq. Signal d'achat à s < −1,25 (seuil du papier), watch à −1, sortie à s > −0,5.
- Backtest walk-forward quotidien sur ~2 ans (chaque jour n'utilise que les 60 j précédents).

## 3. Valorisation relative (overlay VALUE) — régressions à la Damodaran
- Régressions de marché avec effets fixes sectoriels :
  log(P/E forward) ~ croissance BPA + ROE + bêta + payout ;
  log(EV/EBITDA) ~ croissance CA + marge EBITDA + bêta (hors financières) ;
  log(P/B) ~ ROE + bêta (banques, assurance, services financiers).
- Résidu = décote/prime non expliquée par les fondamentaux ; value_z < 0 = bon marché.
  Un signal statistique sur un titre cher en fondamentaux est signalé `rich_on_fundamentals`.

## 4. Overlay macro quantitatif
- Facteurs hebdo (yfinance) : Euro Stoxx 50 (contrôle), VSTOXX, taux US 10 ans, EUR/USD, Brent, gaz TTF, cuivre, or.
- Régression ridge de chaque titre sur les mouvements standardisés des facteurs (104 semaines).
- Régime = mouvement de chaque facteur sur 4 semaines en écarts-types. Score macro = Σ bêta × régime
  (hors marché) ; exprimé en percentile de l'univers.
- La tâche Claude hebdo ajoute la lecture qualitative (BCE, inflation, PMI, actualité société) et un verdict
  favorable / neutre / défavorable par idée.

## 5. Score composite (0-100)
30 % force du signal, 25 % qualité statistique (dont backtest), 20 % valorisation, 20 % macro, 5 % rendement/risque.
Pénalités : chute > 15 % sur 20 j (×0,85), résultats sous 10 j (×0,92). Bonus +3 par signal indépendant
confirmant le même titre (max +9).

## Niveaux
TP = retour complet de l'écart à l'équilibre (autre jambe inchangée). SL = max(1,5 σ_eq, 1,25 × vol journalière × √demi-vie).
Horizon = 2 × demi-vie (5 à 60 jours de bourse).

## Limites
Tests multiples (des centaines de paires) : à 5 % on attend des faux positifs, d'où la double fenêtre et le backtest.
Les rendements des stratégies de paires ont fortement baissé depuis les années 2000. Données yfinance gratuites,
fondamentaux parfois manquants ou en retard. Recherche/éducation, pas un conseil en investissement.

Références : Gatev, Goetzmann & Rouwenhorst (2006, RFS) ; Avellaneda & Lee (2010, Quantitative Finance) ;
Engle & Granger (1987) ; Damodaran, *The Little Book of Valuation*.
