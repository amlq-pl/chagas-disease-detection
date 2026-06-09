# chagas-disease-detection

## Wymagania
`requirements.txt`, jak na razie:

```
numpy>=1.21
pandas>=1.3
scipy>=1.7
```

## Pobieranie danych i info o modelach
### PTB-XL - dane zdrowych osób
https://zenodo.org/records/4905618
najlepiej zmień sobie nazwę folderu na ptbxl-data żeby nie musieć dodawać flag na input
### Sami-Trop - dane chorych osób
https://zenodo.org/records/4905618
wrzuć oba pliki ze strony do folderu samitrop-data
### CODE-15% - dużo danych z weak labelami
https://zenodo.org/records/4916206

Do CODE-15% jeszcze nie dodałem skryptu (ale pobierz je sobie w wolnej chwili, bo są duże), na razie skupmy
się i tak na tych pozostałych dwóch.
Z modeli na pewno chcemy mieć ResNeta 1D, oprócz tego jeszcze kilka innych fajnie by było napisać jutro (może po 2 na głowę?)
Ja postaram się napisać spektrogram  na podstawie tych danych i jakiś ResNet 2D, ale jak chcesz to przejąć to daj znać - ogólnie
pisz na messengerze, co developujesz, żebyśmy się nie zabrali za to samo i fajnie byłoby zachować różnorodność i uwzględnić
w projekcie różne typy modeli. Potencjalnie te, które będą fajnie działać możemy jakoś spróbować połączyć, zresztą zobaczymy jak to w ogóle będzie działać XD

A i przy wyborze modeli jeszcze ważne jest to, żeby dobrze sobie radziły z przesuniętymi danymi - w sensie dla każdego recordu
dla t=0, może być inna faza EKG - np. sieci konwolucyjne sobie fajnie to obsługują, a MLP chyba nie.

No i ważne, że mamy dość mało danych o pozytywnych labelach - zwróć na to uwagę podczas pisania kodu.

Wiadomo, że docelowo mid fusion z metadanymi (których jest mało), ale najpierw najważniejsze, żeby mieć zaimplementowane po prostu uczenie na samym EKG - metadane dodamy później.

Jak na razie nie przejmujmy się tym, że jakieś funkcje można zunifikować w przyszłości (typu wyniki uczenia itd.) - jak starczy czasu to zrobimy to na końcu dla przejrzystości.

A i w sumie fajnie gdybyś dodawał jakieś komentarze po polsku - zmatchuj konwencję ze skryptów, żebym mogł jakoś przyjemnie
przejść przez ten kod i wiedział, co tam się dzieje.

I MEGA WAŻNE: to, że skuteczność będzie średnia nie jest samo w sobie jakoś istotne, zwłaszcza, że labelów pozytywnych jest mało.
Najważniejsze jest to, żeby poprawnie wykrywać pozytywne (hehe) wyniki, bo na tym polegają takie modele, żeby odsiewać
na pewno zdrowe osoby i robić dodatkowe badania tym potencjalnie chorym.

## Skrypty
Ogólnie pobierz sobie dane i użyj skryptów:
 - `prepare_ptxbl_data.py`
 - `prepare_samitrop_data.py`
 - `merge_and_process_data.py`
Więcej info masz samych tych plikach.
SamiTrop miał ogólnie na moduł większe wartości, więc zrobiłem standaryzację.

## Opis planu działania na potem
Na podstawie modeli, będziemy labelować te duże dane CODE-15% - to będzie sporo pracy, dlatego dziś (we wtorek) musimy się spiąć,
żeby mieć ogarnięte, to co napisałem wyżej
najwyżej w środę też wezmę wolne.

I ogólnie to dość ambitne, ale sytuacja jest taka:
jak wybierzemy model, który najfajniej handluje EKG, to wdrażamy etap 2:

Duże dane to weak labele: to znaczy najpewniej pozytywne labele są prawdziwe, bo objawy ciężko pomylić/zmyślić - zresztą
nie wierzę, że lekarze nie mieli wpływu na te labele, patrząc po objawach. Ale sporo negatywnych labeli może mieć chorobę
tylko bezobjawowo. Więc możemy iteracyjnie zwiększać nasz model - trzeba uważać tutaj na powielanie błędów modelu, potencjalnie
artykuł, który ci wysłałem w niedzielę to rozwiązuje. Będziemy sami dopasowywać labele do danych i jeśli będą graniczyły
z pewnością to dodajemy do naszego databasu. Wtedy też możemy wdrożyć kolejne metadane - one się bardzo mogą przydać, zwłaszcza
biorąc pod uwagę obecność danych powiązanych z sercem. Jest również jedno wspólne pole - death i timey wspólne z SamiTropem, więc
to też na plus.

Sory za chaotyczność ale jest 5 w nocy/rano xd

## Kroki działań
1. Piszemy modele dla pierwszych dwóch badań
2. Dodajemy mid fusion z wiekiem + płcią
3. Porównujemy wyniki - wybór najlepszego - potencjalna fuzja?
4. Wdrażamy wybrany model do największego badania po samym EKG - na podstawie opisu wyżej
5. Jak starczy czasu, dodajemy do tego modelu jakiegoś XGBoosta na podstawie metadanych z tego największego badania + death i timey z Samitrop, jeszcze dałoby się skorzystać z gotowego modelu do przewidywania wieku na podstawie EKG (to dałoby info, jaki tryb życia ktoś prowadzi), ale to już chyba overkill i nie starczy czasu pewnie XD. tak czy siak link do tego: https://github.com/antonior92/ecg-age-prediction
