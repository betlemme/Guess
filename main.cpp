#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/nonfree/nonfree.hpp"    //importante per surf e sift

#include <iostream>
#include <fstream>
#include <string>     // std::string, std::to_string

using namespace std;
using namespace cv;

//legge i nomi delle immagini dal file "trainImages.txt"
static void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == String::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

//popola il vettore di immagini di training "trainImages"
static bool readImages( const string& trainFilename,
                 vector <Mat>& trainImages, vector<string>& trainImageNames )
{
    cout << "< Reading the images..." << endl;

    string trainDirName;
    readTrainFilenames( trainFilename, trainDirName, trainImageNames );
    if( trainImageNames.empty() )
    {
        cout << "Train image filenames can not be read." << endl << ">" << endl;
        return false;
    }
    int readImageCount = 0;
    for( size_t i = 0; i < trainImageNames.size(); i++ )
    {
        string filename = trainDirName + trainImageNames[i];
        Mat img = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
        if( img.empty() )
            cout << "Train image " << filename << " can not be read." << endl;
        else
            readImageCount++;
        trainImages.push_back( img );
    }
    if( !readImageCount )
    {
        cout << "All train images can not be read." << endl << ">" << endl;
        return false;
    }
    else
        cout << readImageCount << " train images were read." << endl;
    cout << ">" << endl;

    return true;
}


//estrae le surf dalle immagini campione e le salva nel file "trainDescriptors.yml"
int saveTrainDescriptors()
{
    cv::initModule_nonfree();  //importante per sify e surf

    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;

    featureDetector = FeatureDetector::create( "SURF" );
    descriptorExtractor = DescriptorExtractor::create( "SURF" );

    vector<Mat> trainImages;
    vector<string> trainImagesNames;
    readImages( "trainImages.txt", trainImages, trainImagesNames );

    vector<vector<KeyPoint> > trainKeypoints;
    cout << endl << "< Extracting keypoints from images..." << endl;
    featureDetector->detect( trainImages, trainKeypoints );

    vector<Mat> trainDescriptors;
    cout << "< Computing descriptors for keypoints..." << endl;
    descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

    //per debug stsampo il numero di descrittori
    int totalTrainDesc = 0;
    for( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
        totalTrainDesc += tdIter->rows;

    cout  << "Total train descriptors count: " << totalTrainDesc << endl;
    cout << ">" << endl;

    FileStorage fs("trainDescriptors.yml", FileStorage::WRITE);
    //write(fs, "descriptors_1", tempDescriptors_1);
    fs << "numberOfImages" << (int) trainDescriptors.size();

    for( int i=0; (int) i<trainDescriptors.size(); i++ )
    {
        //per convertire i da int a stringa mi tocca fare tutto sto casin:
        stringstream ss;
        ss << i;
        string nome = ss.str();
        string str = "image_" + nome;
        write(fs, str, trainDescriptors[i]);

    }

    fs.release();



    return 0;
}



//calcola i descrittori dell'immagine di query
Mat findQueryDescriptors(string queryImageName)
{
    Mat img = imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);

    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;

    featureDetector = FeatureDetector::create( "SURF" );
    descriptorExtractor = DescriptorExtractor::create( "SURF" );

    vector<KeyPoint> queryKeypoints;
    cout << endl << "< Extracting keypoints from image query..." << endl;
    featureDetector->detect( img, queryKeypoints );

    Mat queryDescriptors;
    cout << "< Computing descriptors for keypoints..." << endl;
    descriptorExtractor->compute( img, queryKeypoints, queryDescriptors );
    cout << "Query descriptors count: " << queryDescriptors.rows << endl;

    return queryDescriptors;

}


int main()
{
    //chiamo saveTrainDescriptors per debug, in realtà lo chiamo fuori dall'applicazione,
    //e leggo direttamente il file yml contenente i descrittori    
    saveTrainDescriptors();

    Ptr<DescriptorMatcher> descriptorMatcher;
    descriptorMatcher = DescriptorMatcher::create( "FlannBased" );

    //calcolo i descrittori della immagine query
    Mat queryDescriptors = findQueryDescriptors("1.jpg");

    cout << "pri" <<endl;

    // TO-DO: caricare i descrittori salvati nel file "trainDescriptors.yml"
    FileStorage fs2("trainDescriptors.yml", FileStorage::READ);
    //fs2.open("trainDescriptors.yml", FileStorage::READ);

    int itNr;
    fs2["numberOfImages"] >> itNr;

    cout << itNr <<endl;

    vector<Mat> trainDescriptors;
    Mat tmp;
    for (int i=0; (int) i<itNr; i++)
    {
        //per convertire i da int a stringa mi tocca fare tutto sto casin:
        stringstream ss;
        ss << i;
        string nome = ss.str();
        cout << i <<endl;
        string str = "image_" + nome;

        fs2[str] >> tmp;
        trainDescriptors.push_back( tmp );
    }

    //il tickmeter sttampa i trmpi, per debug
    TickMeter tm;

    tm.start();
    descriptorMatcher->add( trainDescriptors ); //qui gli passo le features delle immagini training
    descriptorMatcher->train();     //qui faccio il training
    tm.stop();
    double buildTime = tm.getTimeMilli();

    tm.start();
    vector<DMatch> matches;
    descriptorMatcher->match( queryDescriptors, matches );  //qui faccio il confronto
    tm.stop();
    double matchTime = tm.getTimeMilli();

    CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

    cout << "Number of matches: " << matches.size() << endl;
    cout << "Build time: " << buildTime << " ms; Match time: " << matchTime << " ms" << endl;
    cout << ">" << endl;

    cout << matches[0].imgIdx << endl;

    /*
    for(int i=0; (int) i<matches.size(); i++)
    {
       int y = (int)  matches[i].imgIdx;
       cout << y << endl;
    }

*/

    




    cout << "Hello World!" << endl;
}

