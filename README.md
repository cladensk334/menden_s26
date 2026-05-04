# Menden_S26 - Zeitreiheanalyse 

## Projekt Überblick
Wir analysieren Wetterdaten des Deutschen Wetterdienstes (DWD).
Die Daten wurden von der offiziellen DWD-Website bezogen: https://www.dwd.de/DE/leistungen/cdc/cdc_ueberblick-klimadaten.html

**Kurs:** Zeitreiheanalyse  
**Gruppen Mitglieder:** Clara Denecke, Jonas Müller, Kenia Eguez  
**Semester:** S26


**Datenaufteilung:**
Clara -  Temperatur, Jonas - Luftdruck, Kenia - Dampfdruck und relative feuchte 

**Branches:**
main, temperatur, luftdruck, dampfdruck 

**Temperatur Startionsdaten:** 
| Stations_id | von_datum | bis_datum | Stationshoehe | geoBreite | geoLaenge | Stationsname | Bundesland | Abgabe |
|---|---|---|---|---|---|---|---|---|
| 05703 | 19510101 | 19860131 | 170 | 49.7964 | 9.8949 | Würzburg/Main | Bayern | Frei |
| 05705 | 19470101 | 20260502 | 268 | 49.7704 | 9.9576 | Würzburg | Bayern | Frei |
| 05707 | 18800101 | 19530131 | 175 | 49.7995 | 9.9280 | Würzburg (Physikalisches-Institut) | Bayern | Frei |

-> Wir entscheiden uns für "Würzburg": Daten gehen bis 2026 und sind nicht historisch
