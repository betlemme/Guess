# Guess
E.C. : Guess.GoogleGuess(String percorsoImg) cerca con google reverse image search la foto salvata in “percorsoImg” e ritorna come stringa la guess di google, e salva l’immagine corrispondente alla guess.
-(dipendenze: Jsoup API, Cloudinary API) librerie.zip da importare: http://www.megafileupload.com/1JP7/librerie.zip
-di default carica un immagine chiamata “cat.jpg” e salva l’immagine guess nello stesso percorso del codice


#main.cpp

copiato dall'esempio nella cartrella samples di opencv.
contiene:


-saveTrainDescriptors()
Legge il file "trainImages.txt" contenente il nome delle immagini di train. Estrae le surf dalle immagini campione e le salva nel file "trainDescriptors.yml".
Stavo pensando che nell'app non includiamo questo metodo, ma gli passiamo direttamente il file yml contenente le features già fatto.
("static void readTrainFilenames(...)" e "static bool readImages(...)" sono metodi ausiliari per leggere dal file trainImages.txt)


-Mat findQueryDescriptors(string queryImageName)
calcola i descrittori dell'immagine di query e ritorna la matrice dei descrittori.


-main()
chiama "findQueryDescriptors" sull'immagine da comparare, legge il file yml e si carica tutti i descrittori, e quindi fa il match chiamando : descriptorMatcher->match( queryDescriptors, matches );
matches è un oggetto vector<DMatch> che contiene per ogni feature dell'immagine query la feature dell'immagine di train che ha matchato.
Facendo: matches[i].imgIdx 
si può vedere a quale immagine di train appartiene la feature che ha matchato con l'i-esima feature dell'immagine query.





