# data-analytics-projekt-ws2526

## Einführung 
Dieses Repository dient zum Deployment eines studentischen Projektes im Modul "Data Analytics". Das Projekt befasst sich mit der Anwendung analytischer Methoden auf einen Datensatz, der "Yellow Taxi Trip Data" aus dem Jahre 2023 (Januar) umfasst. Das Deployment bezieht sich auf das Streamlit-Dashboard, das den finalen Teil des CRISP-DM Zyklus finalisieren soll. Somit soll es Kommilitonen, Professoren oder externen Interessenten möglich gemacht werden, die Anwendung zu begutachten. 

## Live-Demo
**[Dashboard aufrufen](https://taxi-nyc-predictor.streamlit.app)** 

## Bestandteile des Dashboards
Das Dashboard ermöglicht die Vorhersage von Fahrtpreisen und Trinkgeldern für NYC Yellow Taxis. Es basiert auf zwei trainierten Machine Learning Modellen:

- **Fare Model**: Vorhersage des Fahrtpreises basierend auf Route, Distanz, Tageszeit und Verkehrslage
- **Tip Model**: Vorhersage des Trinkgelds unter Berücksichtigung von Wetterdaten

Der Nutzer wählt Pickup- und Dropoff-Location, Datum/Uhrzeit sowie Passagieranzahl – das Dashboard liefert dann die Vorhersagen inkl. historischem Vergleich.

## Projektstruktur
```
├── app.py                 # Streamlit Dashboard
├── requirements.txt       # Python-Abhängigkeiten
├── utils/                 # Hilfsfunktionen (Haversine-Distanz)
├── notebooks/             # Jupyter Notebooks (Modelltraining)
├── models/final/          # Lokale Modellkopien
├── data/                  # Taxi-Zonen & Routenstatistiken
└── weather/               # Wetterdaten Januar 2023
```

## Datenquellen & Modelle
Die trainierten Modelle sowie alle benötigten Daten werden zur Laufzeit von **Hugging Face** geladen:
- Repository: [dnltre/taxi-nyc-models](https://huggingface.co/dnltre/taxi-nyc-models)

Dadurch ist keine lokale Datenhaltung erforderlich – das Dashboard funktioniert direkt in der Cloud.

## Lokale Ausführung
Falls gewünscht, kann das Dashboard auch lokal gestartet werden:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Technologien
- Python, Pandas, Scikit-learn
- Streamlit (Dashboard)
- Hugging Face Hub (Modell-Hosting)

---
*Hochschule für Technik Stuttgart – WiSe 2025/26* - *Master Digitale Prozesse & Technologien*