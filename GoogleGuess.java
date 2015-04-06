import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.File;
import java.util.Map;
import java.net.URL;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.cloudinary.*;
import com.cloudinary.utils.ObjectUtils;

 
public class GoogleGuess {
	
	public static void main(String[] args) throws IOException{
		System.out.println(guess("cat.jpg"));
		
	}
	 
	public static String guess(String percorsoFile) throws IOException {
		String str = percorsoFile;
		File toUpload = new File(str);
		
		//Posto la foto e ottengo l'url:
		Cloudinary cloudinary = new Cloudinary(ObjectUtils.asMap(
				  "cloud_name", "jpegembedded",
				  "api_key", "652944918467885",
				  "api_secret", "NGXgbYVSw6sqUu_tFaJIzBvrL8c"));
		
		Map uploadResult = null;
		try {
			uploadResult = cloudinary.uploader().upload(toUpload, ObjectUtils.emptyMap());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("non riesco a fare l'upload dell'immagine");
			e.printStackTrace();
		}
		String imgUrl = (String) uploadResult.get("url");
		System.out.println(imgUrl);
		
		//Costruisco l'url per cercare l'immagine su google immagini:
		String google2 = "https://www.google.it/";
		String google = "https://www.google.com/searchbyimage?&image_url=";
		//String imgUrl = "http://res.cloudinary.com/jpegembedded/image/upload/v1428322006/sbpkfgg331kgzhgd429g.jpg";
		String url = google+imgUrl;
		System.out.println(url);
		
		//cerco il best guess:
		//scarico la pagina html dei risultati (è reindirizzata, ma jsoup segue bene il redirect):
		Document doc = null;
		try {
			doc = Jsoup.connect(url).userAgent("Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.76 Safari/537.36").timeout(5000).get();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("google ci mette troppo a tornarmi la pagina dei risultati");
			e.printStackTrace();
		}
		//trovo il nodo che racchiude la stringa del best guess (classe _gUb, trovata ispezionando a mano la pagina sorgente):
		Elements bestguess = doc.select("a._gUb");
		//ricavo la stringa
		String best = bestguess.text();
		System.out.println(best);
		
		//ora ricavo dal doc l'url della pagina contenente tutte le immagini simili:
		//il nome della classe è stato ottenuto sempre ispezionando a mano il doc sorgente
		Elements linkBestImgs = doc.select("div._Icb._kk._wI > a");
		String link = linkBestImgs.attr("href");
		String url2 = google2+link;
		System.out.println(url2);

		//scarico la pagina dei risultati simili:
		Document doc2 = null;
		try {
			doc2 = Jsoup.connect(url2).userAgent("Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.76 Safari/537.36").timeout(5000).get();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("ci metto troppo a raggiungere url2");
			e.printStackTrace();
		}
		//ispeziono il primo risultato e ricavo il link all'immagine:
		Element masthead = doc2.select("#rg_s > div:nth-child(1) > a").first();
		String mastheadLinkString = masthead.attr("href");
		System.out.println(mastheadLinkString);
		//ora devo parsare la stringa dal carattere successivo al primo '=' fino alla prima occorrenza di '.jpg' inclusa
	    int indexname = mastheadLinkString.indexOf("=");
	    indexname++;
	    int indexname2 = mastheadLinkString.indexOf("&");
	      
	    mastheadLinkString = mastheadLinkString.substring(indexname, indexname2);
	    System.out.println(mastheadLinkString);

		

		/*	non conviene pescare da qui perchè non sempre è presente questo div	
		System.out.println("waiting for img a");
		
		//vado all'url trovato e ricavo la prima immagine:
		//#R6BLeEptk3nUzM\3a  .bia.uh
		//div.bicc
		Elements bestFoto = doc.select(".bia.uh_rl");
		for(Element e : bestFoto){
			System.out.println(e.attr("href"));
		}
		
		
		System.out.println("nkdsajk");
		 */	
	    
	    //infine salvo l'immagine:
	      
			URL indirizzo = new URL(mastheadLinkString);
			InputStream is = indirizzo.openStream();
			OutputStream os = new FileOutputStream("image.jpg");

			byte[] b = new byte[2048];
			int length;

			while ((length = is.read(b)) != -1) {
				os.write(b, 0, length);
			}

			is.close();
			os.close();
		

			return best;
 
	}
 
}
