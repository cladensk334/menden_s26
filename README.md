# Menden_S26 - Zeitreiheanalyse 

## Projekt Überblick
Wir analysieren Wetterdaten des Deutschen Wetterdienstes (DWD).
Die Daten wurden von der offiziellen DWD-Website bezogen: https://www.dwd.de/DE/leistungen/cdc/cdc_ueberblick-klimadaten.html

**Kurs:** Zeitreiheanalyse  
**Gruppen Mitglieder:** Clara Denecke, Jonas Müller, Kenia Eguez  
**Semester:** S26

**Welche Dateien gibt es?**
data (order): Rohdaten und bereinigte Daten beinhaltet, ToDo: was noch gemacht werden muss, src: notwendige Python Funktionen, Aufgabenstellung: unsere Project eckdaten 

**Wie klont man dieses Respiratory lokal ?**

1. Geh auf GitHub zum Repository
2. Klick auf den grünen Button Code
3. Wähl den Reiter SSH
4. Kopier die Adresse — sie sieht so aus:
  git@github.com:IhrBenutzername/IhrRepository.git
5. im Terminal folgenden Befehl ausführen:
  bashgit clone git@github.com:IhrBenutzername/IhrRepository.git
6. Danach in den Ordner wechseln:
  bashcd IhrRepository


**Datenaufteilung:**
Clara -  Temperatur, Jonas - Luftdruck, Kenia - Wind

**Branches:**
main (mit Ordnerstrukur), temperatur , luftdruck, wind (hier alle Unterteilt in die jeweiligen Arbeitsschritte), neuner Branch der die Modelle der drei einzelnen  zusammenführt und jeweils die besten Schritte nimmt 

**Regeln:**
1. Jeder Uplaod wird direkt in den richtigen Ordner gesteckt (hier ist die Strucktur Daten(raw/processed), notebooks
2. Whatsapp Gruppe zur Kommunikation plus 1-2 Meetings pro Woche nach Aufgabenverteilung
3. Bei Struktur arbeiten (Bsp neuer Branch/Ordner/ ect) gegenseiting Informieren um Fehler zu vermeiden
4. Der Main Branch wird nur am Anfang (zb Upload der Daten, Read me) benutzt, danach der spezifische Aufgaben Branch
5. Vor einem upload muss die aktuelle Version von GitHub runtergeladen werden und diese angepasst werden, um Konfikte zu vermeiden
6. Code (oder andere nicht eindeutige Dateien) sollten mit Kommentar hochgeladen werden um verwirrung zu vermeiden

**Was macht ein professionelles Repo aus ?:**
- so aufgebaut, dass jemand Fremdes es klonen, verstehen und sofort ausführen kann ohne Rückfragen zu haben
- aussagekräftiges README
- saubere Ordnerstruktur
- .gitignore damit keine unnötigen Dateien getrackt werden (können wir hier nicht machen)
- requirements.txt damit die Umgebung reproduzierbar ist
- Commits die eine nachvollziehbare Geschichte des Projekts erzählen
- Hier findet man weitere Ideen: https://github.com/orgs/community/discussions/88715

**Temperatur Stationsdaten:** 
| Stations_id | von_datum | bis_datum | Stationshoehe | geoBreite | geoLaenge | Stationsname | Bundesland | Abgabe |
|---|---|---|---|---|---|---|---|---|
| 05703 | 19510101 | 19860131 | 170 | 49.7964 | 9.8949 | Würzburg/Main | Bayern | Frei |
| 05705 | 19470101 | 20260502 | 268 | 49.7704 | 9.9576 | Würzburg | Bayern | Frei |
| 05707 | 18800101 | 19530131 | 175 | 49.7995 | 9.9280 | Würzburg (Physikalisches-Institut) | Bayern | Frei |

-> Wir entscheiden uns für "Würzburg": Daten gehen bis 03.05.2026 und sind nicht historisch
