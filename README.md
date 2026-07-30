# Black-Box — Desk de Trading Algorithmique Systématique

Fondations d'une black box de trading quantitatif : ingestion de données,
génération d'alpha (statistique + ML), gestion du risque et exécution,
pilotable et supervisable via Discord. Calibré sur les standards de
*Inside the Black Box* (Narang), *Advances in Financial Machine Learning*
(López de Prado) et *Quantitative Trading* (Chan).

## 1. Choix de l'infrastructure

| Couche | Choix | Justification |
|---|---|---|
| Recherche & ML | **Python 3.11+** (pandas, numpy, scikit-learn, statsmodels) | Écosystème de référence pour le feature engineering, la validation croisée purgée et le prototypage rapide de signaux (De Prado). |
| Exécution critique en latence | Python asyncio, avec point d'extension **C++/Rust** | Pour ce socle (fréquence bar/minute sur actions/ETF), Python asyncio suffit. Un desk visant du market-making ou du HFT isolerait le chemin *order routing* dans un service C++/Rust communiquant par message queue (ZeroMQ/gRPC) — prévu comme extension, non nécessaire au stade actuel. |
| Stockage historique | **PostgreSQL** (extension TimescaleDB recommandée en prod) | Durabilité, requêtes SQL pour la recherche, partitionnement temporel naturel pour des séries OHLCV. |
| État temps réel / contrôle | **Redis** | Latence sub-milliseconde pour le dernier prix, les positions courantes, et surtout le **flag kill-switch** partagé entre le bot Discord et le moteur — aucun composant ne doit dépendre d'un appel bloquant pour savoir s'il doit s'arrêter. |
| Broker | **Alpaca** (equities/ETF, paper & live) via adaptateur `BrokerAPI`, extensible à **Interactive Brokers** | API REST simple, environnement paper trading gratuit pour valider l'infrastructure avant capital réel. L'abstraction `BrokerAPI` permet de brancher IBKR sans toucher au reste du moteur. |
| Pilotage / alerting | **Discord** (webhook + bot de commandes slash) | Alertes de risk management, snapshots PnL, et commandes `/status`, `/kill`, `/resume` accessibles depuis un téléphone — aucune dépendance à un accès SSH pour déclencher un arrêt d'urgence. |

Le principe directeur (Narang, ch.2) : chaque couche communique avec la
suivante via une interface étroite et remplaçable (`MarketDataProvider`,
`BrokerAPI`), jamais par couplage direct — un changement de broker ou de
fournisseur de données ne doit jamais se propager dans la logique d'alpha
ou de risque.

## 2. Architecture de la Black Box

```
                        ┌─────────────────────────────────────────┐
                        │              DISCORD (control plane)     │
                        │  Webhook: alertes risque, PnL, fills      │
                        │  Bot: /status /kill /resume  ──┐          │
                        └─────────────────────────────────┼──────────┘
                                                           │ (Redis pub/sub)
┌──────────────┐      ┌──────────────┐      ┌─────────────▼──────────┐      ┌──────────────┐
│  1. DONNÉES   │ ───► │  2. ALPHA     │ ───► │  3. RISQUE              │ ───► │ 4. EXÉCUTION │
│              │      │              │      │                         │      │              │
│ MarketData-  │      │ Features      │      │ RiskManager             │      │ BrokerAPI     │
│ Provider     │      │ (frac-diff,   │      │  - vol targeting        │      │  (Paper /     │
│  (yfinance/  │      │  z-score,     │      │  - Kelly fractionnaire  │      │   Alpaca /    │
│   Alpaca)    │      │  ATR, RSI)    │      │  - stop ATR dynamique   │      │   IBKR)       │
│              │      │              │      │  - limites drawdown     │      │              │
│ DataCleaner  │      │ Signaux:      │      │    (jour / total)       │      │ OrderManager  │
│  (outliers,  │      │  - Mean-Rev   │      │  - kill-switch          │      │  - netting     │
│   gaps)      │      │    (z+ADF)    │      │                         │      │  - flatten     │
│              │      │  - Momentum   │      │                         │      │    d'urgence   │
│ Postgres     │      │    (MA cross) │      │                         │      │              │
│  (historique)│      │              │      │                         │      │              │
│ Redis        │      │ Meta-labeling │      │                         │      │              │
│  (temps réel)│      │  ML (RF +     │      │                         │      │              │
│              │      │  PurgedKFold) │      │                         │      │              │
└──────────────┘      └──────────────┘      └─────────────────────────┘      └──────────────┘
```

### Module 1 — Données (`blackbox/data/`)
- `market_data.py` : interface `MarketDataProvider` abstraite ;
  implémentations `YFinanceProvider` (recherche/backtest) et
  `AlpacaDataProvider` (production).
- `DataCleaner` : dédoublonnage, gestion des trous, winsorization des
  rendements aberrants (z-score des rendements > 8σ), forward-fill des
  OHLC.
- `bars.py` : bars dollar (agrégation par volume notionnel échangé,
  De Prado ch.2) et volatilité journalière EWMA, utilisées comme largeur
  de barrière dans le labeling.
- `storage.py` : `PostgresStore` (historique durable) et `RedisStore`
  (équity, positions, kill-switch, pub/sub commandes).

### Module 2 — Alpha (`blackbox/alpha/`)
- `features.py` : différenciation fractionnaire à fenêtre fixe (mémoire
  longue + stationnarité, AFML ch.5), z-score glissant, ATR, RSI,
  volatilité réalisée annualisée — tout calculé sans fuite d'information
  future.
- `signals.py` :
  - **Mean Reversion** : z-score sur fenêtre glissante, filtré par un
    test de stationnarité ADF (on ne trade la réversion que sur des
    régimes statistiquement mean-reverting, Chan ch.2).
  - **Momentum** : croisement de moyennes mobiles rapide/lent, force
    normalisée par la moyenne lente.
- `labeling.py` : méthode des **trois barrières** (profit-take /
  stop-loss / barrière temporelle, AFML ch.3) et **meta-labeling** —
  un signal primaire donne la direction, un label binaire indique si le
  trade aurait été gagnant.
- `model.py` : `PurgedKFold` (validation croisée qui purge les
  observations d'entraînement chevauchant la fenêtre de label du test,
  plus embargo — AFML ch.7) et `MetaLabelingModel` (Random Forest)
  produisant une **probabilité de confiance** utilisée pour le
  dimensionnement, pas pour remplacer le signal primaire.
- `candles.py` + `PriceActionConfluenceSignal` (dans `signals.py`) :
  troisième régime, en repli quand mean-reversion et momentum sont
  silencieux — filtre de tendance multi-timeframe **calculé par
  resampling réel de la même série** (pas une série simulée
  indépendamment) et décalé d'une bougie HTF pour ne référencer que des
  bougies déjà clôturées (`features.higher_timeframe_trend`), croisé
  avec une extrémité RSI/Bollinger et un pattern de bougie confirmant
  (doji, engulfing). Le score de confluence est **continu** (construit
  à partir de la profondeur RSI/BB au-delà du seuil), pas une constante
  arbitraire.

  > Ce module corrige un template externe qui simulait ses données
  > (`np.random`), calculait un filtre "multi-timeframe" à partir de
  > deux séries indépendamment simulées avec le même seed (donc sans
  > lien réel avec le sous-jacent tradé), affichait un score de
  > confiance codé en dur, et — en l'absence de signal — dispatchait un
  > trade fictif indiscernable d'une alerte réelle sur le même canal
  > Discord. La version Pine Script associée souffrait en plus d'un
  > repaint classique sur son filtre MTF (`request.security` sans
  > référence à la bougie HTF confirmée) et d'un sizing fixe déconnecté
  > de la distance du stop. Voir `examples/demo_confluence_alert.py`
  > (données réelles, aucun signal fictif dispatché) et
  > `pine/institutional_confluence_strategy.pine` (MTF non-repeint,
  > coûts modélisés, sizing basé sur le risque).

### Module 3 — Gestion des Risques (`blackbox/risk/`)
- `position_sizing.py` : sizing par ciblage de volatilité (exposition
  dont la vol standalone correspond à une cible annualisée) et **Kelly
  fractionnaire plafonné** (demi-Kelly par défaut), modulé par la
  probabilité de confiance du meta-modèle.
- `risk_manager.py` :
  - Stop-loss et take-profit **dynamiques basés sur l'ATR** (plus
    larges en régime volatil, Chan ch.3), calculés symétriquement à
    partir de la même estimation de volatilité — pas de cible fixe en
    pips/pourcentage déconnectée du régime courant.
  - Limites de **drawdown journalier et total** déclenchant un
    **kill-switch** automatique (flatten immédiat + alerte Discord).
  - Plafond de levier brut et de taille maximale par position,
    appliqué indépendamment de ce que recommande le sizing.

### Module 4 — Exécution (`blackbox/execution/`)
- `broker.py` : interface `BrokerAPI` ; `PaperBroker` (simulation avec
  slippage + commissions pour ne jamais halluciner un fill parfait) et
  `AlpacaBroker` (ordres marché réels, liquidation totale d'urgence).
- `order_manager.py` : traduit une position cible risk-approved en un
  ordre net (pas d'empilement d'ordres), et expose le chemin de
  **flatten d'urgence** utilisé par le kill-switch.

### Boucle de décision (`blackbox/core/engine.py`)
À chaque itération : vérifie le kill-switch (local + Redis) → pour
chaque symbole de l'univers, récupère l'historique → calcule signaux et
confiance ML → dimensionne et clippe la position via le risk manager →
envoie l'ordre net → met à jour l'équity/drawdown → notifie Discord.

### Backtesting (`blackbox/backtest/backtester.py`)
Backtest vectorisé avec **coûts de transaction réalistes** (commission +
slippage en bps sur chaque changement de position), fenêtres
**walk-forward** en expanding window, et reporting complet (Sharpe,
Sortino, max drawdown, CAGR, turnover) — jamais un rendement brut isolé
(Chan, ch.5).

## 3. Utilisation

```bash
cp .env.example .env          # renseigner les clés broker / Discord si besoin
pip install -r requirements.txt
docker compose up -d          # Postgres + Redis

# Backtest de recherche (mode paper, données yfinance par défaut)
python main.py backtest --symbol SPY --start 2018-01-01

# Boucle live/paper avec kill-switch Discord
python main.py live
```

## Site de test interactif -- Black-Box Scanner

Un dashboard Streamlit (`webapp/app.py`) façon scanner : tape un
symbole, choisis un horizon, clique **Analyser**, et obtiens une
recommandation claire sans jamais toucher un broker.

- **Recherche par nom ou ticker** : tape un nom d'entreprise ("Apple",
  "Broadcom") ou directement un ticker ("AAPL", "AVGO") -- résolu via
  `yfinance.Search`, avec une liste de correspondances si plusieurs
  titres matchent. Un **mode démo hors-ligne** (données synthétiques,
  qui n'appelle pas la recherche) permet de tester le scanner sans
  accès réseau ou marché fermé -- utile aussi pour l'import d'un CSV
  perso (`date, open, high, low, close, volume`).
- **Horizons d'analyse** : 5 min (scalp), 15 min (intraday), 1 heure
  (swing court) ou 1 jour (position). Chaque horizon fixe
  automatiquement la bougie d'entrée et sa confirmation de tendance sur
  un timeframe strictement plus large (ex. 5 min confirmé par 30 min),
  exactement le mapping qu'utilise le moteur live.
- **Carte de recommandation** : badge ACHAT / VENTE À DÉCOUVERT /
  NEUTRE, régime actif du fallback mean-reversion → momentum →
  price-action, horizon de tenue indicatif, entrée/stop/take-profit
  ATR et ratio risque/rendement, jauge de confiance, et le détail
  "pourquoi cette décision" (quel palier a voté quoi).
- **Bougies + RSI + actualisation automatique** : graphique en
  chandeliers OHLC réels avec les points d'entrée de chaque signal, un
  panneau RSI avec seuils survente/surachat, et une option
  d'actualisation automatique (30s) qui recharge les données et
  relance l'analyse toute seule.
- **Paramètres avancés** (repliés par défaut) : réglages stratégie et
  risque, entraînement optionnel du meta-modèle ML (Random Forest +
  PurgedKFold) avec précision OOS et importances des features,
  backtest détaillé (Sharpe/Sortino/max drawdown/CAGR/turnover) et
  aperçu des données brutes.

### Lancement en un clic (sans taper de commande)

`scripts/run_scanner.bat` (Windows), `scripts/run_scanner.command`
(macOS) ou `scripts/run_scanner.sh` (Linux) clonent le dépôt, créent un
environnement virtuel, installent les dépendances et lancent le
scanner -- tout reste local, rien n'est mis en ligne. Télécharge le
script correspondant à ton OS et double-clique dessus (Git et Python 3
doivent être installés au préalable).

```bash
pip install -r requirements-web.txt
streamlit run webapp/app.py
```

## Tests

Suite pytest (55 tests) couvrant : nettoyage de données et bars dollar,
features (RSI, ATR, Bollinger, différenciation fractionnaire, filtre
HTF sans lookahead), reconnaissance de patterns de bougies, triple
barrière + meta-labeling, `PurgedKFold` (absence de fuite train/test),
signaux (mean-reversion, momentum, confluence price action), risk
manager (stop/take-profit ATR, kill-switch drawdown, plafonds
d'exposition), sizing (Kelly, ciblage de volatilité), backtester (coûts
de transaction, fenêtres walk-forward), exécution (`PaperBroker`,
`OrderManager`) et intégration du moteur complet.

```bash
pip install -r requirements-dev.txt
pytest
```

La CI (`.github/workflows/tests.yml`) exécute cette suite sur chaque
push et pull request.

## Avertissement

Ce dépôt est un **socle d'architecture et de recherche**, pas un système
prêt pour la production avec du capital réel. Avant tout déploiement
live : audit de sécurité des credentials, tests de robustesse du
kill-switch en conditions de panne réseau/broker, et validation
walk-forward out-of-sample sur l'univers cible.
