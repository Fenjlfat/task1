// ANN_SPH.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
// NEURONKA_MG.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.


#include <iostream>
#include <stdio.h>
//#include <conio.h>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
using namespace::std;

//параметры нейросети
struct PARAMETRS_ANN
{
    int layer = 4;
    int neuron = 32;
    int connect = 32;
    int input_Data = 6;
    int output_Data = 7;
    int columns_File = 13;
    int function = 2; //function activation 1=sigmoidal, 2=ReLU
    int epochs = 50000;
    int counter = 500;      // show graphycs after number of epochs 
    int BEGIN_batch = 0;//начало батча
    int END_batch = 203;//конец батча
    double tochnost = 0.0004;
    double alpha = 0.01;
    double koef_relu = 0.05;
    double koef_normirovki = 0.01;
    double koef_DEL = 10.00;
    double weightMin = 0.01, weightMax = 0.95;  //initial range of weights
    double biasMin = 0.1, biasMax = 0.14;       //initial range of biases
    string name_File = "SPH_CU_CUBE_V1D5_L5_R18_T300_X.txt";
    string way_to_DataGnuplot = "/mnt/С++/ANN_SPH/DataGnuplot.txt";

    string plot_1 = "1:2";   // TEMP_SPH
    string plot_2 = "1:3";   // ENERGY_SPH
    string plot_3 = "1:4";   // SXX_SPH
    string plot_4 = "1:5";   // SYY_SPH
    string plot_5 = "1:6";   // SZZ_SPH
    string plot_6 = "1:7";   // RHOD_SPH
    string plot_7 = "1:8";   // POROSITY_SPH

    string plot_8 = "1:9";   // TEMP_ANN
    string plot_9 = "1:10";  // ENERGY_ANN
    string plot_10 = "1:11"; // SXX_ANN
    string plot_11 = "1:12"; // SYY_ANN
    string plot_12 = "1:13"; // SZZ_ANN
    string plot_13 = "1:14"; // RHOD_ANN
    string plot_14 = "1:15"; // POROSITY_ANN
};
/*//рисование графиков
double GNUPLOT(int numberPoint, vector<vector<double>>& vector_Output, vector<vector<double>>& vector_Output_ANN, string graph1, string graph2)
{
    PARAMETRS_ANN PARAM_GNUPLOT;
    //string message{ "plot  '" + PARAM_GNUPLOT.way_to_DataGnuplot + "' using " + graph1 + " title 'MD' with lines linecolor 1, '" + PARAM_GNUPLOT.way_to_DataGnuplot + "' using " + graph2 + " title 'ANN' with lines linecolor 5\n" };
    //const char* WAY_TO_DATA = message.c_str();  // преобразуем в указатель

    //FILE* pipe = _popen("C:\\gnuplot\\bin\\gnuplot.exe", "w");
    FILE* pipe = popen("gnuplot -", "w");
    if (pipe != NULL)
    {
        ofstream DATAGNUPLOT;

        DATAGNUPLOT.open("DataGnuplot.txt");
        for (int i = 0; i < numberPoint; i++)
        {
            DATAGNUPLOT << i << " " << vector_Output[i][0] << " " << vector_Output[i][1] << " " << vector_Output[i][2] << " " << vector_Output[i][3] << " " << vector_Output[i][4] << " " << vector_Output[i][5] << " " << vector_Output[i][6]
                << " " << vector_Output_ANN[i][0] << " " << vector_Output_ANN[i][1] << " " << vector_Output_ANN[i][2] << " " << vector_Output_ANN[i][3] << " " << vector_Output_ANN[i][4] << " " << vector_Output_ANN[i][5] << " " << vector_Output_ANN[i][6] << endl;   //SIG11
        }
        DATAGNUPLOT.close();
        fprintf(pipe, "/mnt/С++/ANN_SPH/DataGnuplot.txt");

        fflush(pipe);

        // ожидание нажатия клавиши
        std::cin.clear();
        std::cin.ignore(std::cin.rdbuf()->in_avail());
        std::cin.get();

#ifdef WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }
    else puts("Could not open the file\n");
    //getchar();
    pclose(pipe);
    return 0;
}*/
//рисование графиков
double GNUPLOT(int numberPoint, vector<vector<double>>& vector_Output, vector<vector<double>>& vector_Output_ANN, string graph1, string graph2)
{
	gp = popen("gnuplot -","w");
	    output = fopen("lorenzgplot.dat","w");
	    float t=0.; 
	    float dt=0.01;
	    float tf=30;
	    float x=3.;
	    float y=2.;
	    float z=0.;
	    float k1x,k1y,k1z, k2x,k2y,k2z,k3x,k3y,k3z,k4x,k4y,k4z;
	    fprintf(output,"%f %f %f \n",x,y,z);
	    fprintf(gp, "splot './lorenzgplot.dat' with lines \n");
	/* Ahora Runge Kutta de orden 4 */  
	    while(t<tf)
	    {
		
	    }
	    fclose(gp);
	    fclose(output);
	    return 0;
}
//нормировка переменных
double Normirovka(double X, double Xmin, double Xmax)
{
    PARAMETRS_ANN PARAMETR;
    double s = 0.00;
    double ret = 0.00;

    if (Xmax != Xmin)
    {
        s = (Xmax - Xmin) * PARAMETR.koef_normirovki;
        Xmin = Xmin - s;
        Xmax = Xmax + s;
        ret = (X - Xmin) / (Xmax - Xmin);
    }
    else
    {
        s = Xmax * PARAMETR.koef_normirovki;
        Xmin = Xmin - s;
        Xmax = Xmax + s;
        ret = (X - Xmin) / (Xmax - Xmin);
    }
    if (Xmax == 0 || Xmin == 0)
    {
        ret = 0.00;
    }
    //Xmin = Xmin - s;
    //Xmax = Xmax + s;
    //return (X - Xmin) / (Xmax - Xmin);
    return ret;
}
double Perenormirovka(double Ynorm, double Ymin, double Ymax)
{
    return Ynorm * (Ymax - Ymin) + Ymin;
    //return Ynorm * (Ymax - Ymin - Ymin * 0.2) + Ymin + Ymin * 0.2;

}

// считывем количество точек файла
double FILE_number_Point(int number)
{
    PARAMETRS_ANN PAR;
    int h = 0, l = 0;
    double q = 0;
    fstream FILE_numberPoint;
    //FILE_numberPoint.open("MD_MG_L20.txt", ios::in);
    FILE_numberPoint.open(PAR.name_File, ios::in);
    if (!FILE_numberPoint.is_open())
    {
        cout << "File not open" << endl;
    }
    else
    {
        cout << "File open" << endl;
        while (!FILE_numberPoint.eof())
        {
            while (h < PAR.columns_File)
            {
                FILE_numberPoint >> q;
                h = h + 1;

            }
            h = 0;
            l = l + 1;
            cout << l << endl;
        }
    }
    number = l;
    return number;
}
//считываем входные и выходные данные с файла
double FILE_INPUT_OUTPUT(int numberPoint, vector<vector<double>>& vector_Input, vector<vector<double>>& vector_Output)
{
    int l = 0, h = 0;
    PARAMETRS_ANN PARAM;

    vector<vector<double>> vector_File(numberPoint, vector<double>(PARAM.columns_File));
    fstream FILE;
    //считывам данные с файла
    //FILE.open("MD_MG_L20.txt", ios::in);
    FILE.open(PARAM.name_File, ios::in);
    if (!FILE.is_open())
    {
        cout << "File not open" << endl;
    }
    else
    {
        cout << "File open" << endl;
        while (!FILE.eof())
        {
            while (h < PARAM.columns_File)
            {
                FILE >> vector_File[l][h];
                h = h + 1;
            }
            h = 0;
            l = l + 1;
        }
    }
    for (int i = 0; i < l; i++)
    {
    for (int j = 0; j < PARAM.columns_File; j++)
    {
    cout << "vectorFile" << i << j << "=" << vector_File[i][j] << "      ";
    }
    cout << endl;
    }
    FILE.close();
    //getchar();
    for (int i = 0; i < numberPoint; i++)
    {
        for (int n = 0; n < PARAM.input_Data; n++)
        {
            vector_Input[i][n] = vector_File[i][n];
        }
    }
    for (int i = 0; i < numberPoint; i++)
    {
        for (int n = 0; n < PARAM.output_Data; n++)
        {
            vector_Output[i][n] = abs(vector_File[i][n + PARAM.input_Data]);
        }
    }
    return 0;
}

//нормируем входной и выходной вектор
double NORMIROVKA(int numberPoint, vector<vector<double>>& vector_Input, vector<vector<double>>& vector_Output,vector<double>& min_Input, vector<double>& max_Input, vector<double>& min_Output, vector<double>& max_Output )
{
   
    PARAMETRS_ANN PARAM;
    //определение максимальных и минимальных значений для нормировки
    for (int j = 0; j < PARAM.input_Data; j++)
    {
        min_Input[j] = vector_Input[0][j];
        max_Input[j] = vector_Input[0][j];
    }
    for (int j = 0; j < PARAM.output_Data; j++)
    {
        min_Output[j] = vector_Output[0][j];
        max_Output[j] = vector_Output[0][j];
    }
    for (int i = 0; i < numberPoint; i++)
    {
        for (int j = 0; j < PARAM.input_Data; j++)
        {
            if (vector_Input[i][j] < min_Input[j])
            {
                min_Input[j] = vector_Input[i][j];
            }
            if (vector_Input[i][j] > max_Input[j])
            {
                max_Input[j] = vector_Input[i][j];
            }            
        }
        for (int j = 0; j < PARAM.output_Data; j++)
        {
            if (vector_Output[i][j] < min_Output[j])
            {
                min_Output[j] = vector_Output[i][j];
            }
            if (vector_Output[i][j] > max_Output[j])
            {
                max_Output[j] = vector_Output[i][j];
            }
        }
    }

    for (int i = 0; i < numberPoint; i++)
    {
        for (int j = 0; j < PARAM.input_Data; j++)
        {
            vector_Input[i][j] = Normirovka(vector_Input[i][j], min_Input[j], max_Input[j]);
        }
        for (int j = 0; j < PARAM.output_Data; j++)
        {
            vector_Output[i][j] = Normirovka(vector_Output[i][j], min_Output[j], max_Output[j]);
        }
    }
    return 0;
}

//функция для заполнения массивов весов и смещений
double random(double min, double max)
{
    PARAMETRS_ANN PARAMETR;
    return ((double)(rand()) / RAND_MAX * (max - min) + min) / PARAMETR.koef_DEL;
}

//функция для инициализации весов и смещений
double Initialization_WEIGHT_BIAS(vector<vector<vector<double>>>& WEIGHT, vector<vector<double>>& BIAS)
{
    PARAMETRS_ANN PARAMETR;
    //инициализация весов
    cout << "initialization  weight" << endl;
    for (int i = 0; i < PARAMETR.layer; i++)
    {
        for (int j = 0; j < PARAMETR.neuron; j++)
        {
            for (int k = 0; k < PARAMETR.connect; k++)
            {
                WEIGHT[i][j][k] = random(PARAMETR.weightMin, PARAMETR.weightMax);
                cout << "weight=" << WEIGHT[i][j][k] << "   ";
            }
            cout << endl;
        }
        cout << endl;
    }
    //intializationn bias
    cout << "initialization  bias" << endl;
    for (int i = 0; i < PARAMETR.layer - 1; i++)
    {
        for (int j = 0; j < PARAMETR.neuron; j++)
        {
            BIAS[i][j] = random(PARAMETR.biasMin, PARAMETR.biasMax);
            cout << "BIAS=" << BIAS[i][j] << "   ";
        }
        cout << endl;
    }
    return 0;
    //getchar();

}

//производная сигмоидальной функции
double DIF_sigmoid(double h)
{
    return ((1 / (1 + exp(-h))) * (1 - (1 / (1 + exp(-h)))));
}

//сигмоидальная функция
double sigmoidal(double g)
{
    return (1 / (1 + exp(-g)));
}

//RELU
double RELU(double t)
{
    PARAMETRS_ANN PARAMETR;
    if (t >= 0)
    {
        return t;
    }
    else
    {
        return t * PARAMETR.koef_relu;
    }

}

//Differencial RELU
double DifRELU(double t)
{
    PARAMETRS_ANN PARAMETR;
    if (t >= 0)
    {
        return 1;
    }
    else
    {
        return  PARAMETR.koef_relu;
    }

}


//вычисление сигналов нейронов и выходных значений
double PERCEPTRON(vector<vector<vector<double>>>& WEIGHT, vector<vector<double>>& BIAS, vector<vector<double>>& signalNeurona,
    vector<double>& output, vector<vector<double>>& summaForSigmoid, vector<double>& Input_Array)
{
    double summa = 0.00;
    int z = 0;
    PARAMETRS_ANN PARAMETR;
    //инициализация сигналов входного слоя
    //cout << "initialization input signals" << endl;
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < PARAMETR.neuron; j++)
        {
            for (int k = 0; k < PARAMETR.connect; k++)
            {
                summa += Input_Array[k] * WEIGHT[i][j][k];
            }
            summa = summa + BIAS[i][j];
            summaForSigmoid[i][j] = summa;
            if (PARAMETR.function == 1)
            {
                signalNeurona[i][j] = sigmoidal(summa);
            }
            if (PARAMETR.function == 2)
            {
                signalNeurona[i][j] = RELU(summa);
            }
            summa = 0.00;
        }
    }
    //вычисление сигналов нейронов для скрытого и выходного слоев
    for (int i = 1; i < PARAMETR.layer; i++)
    {
        for (int j = 0; j < PARAMETR.neuron; j++)
        {
            z = PARAMETR.layer - i;
            for (int k = 0; k < PARAMETR.connect; k++)
            {
                summa += signalNeurona[i - 1][k] * WEIGHT[i][j][k];
            }
            summa = summa + BIAS[i][j];
            summaForSigmoid[i][j] = summa;
            if (z == 1)
            {
                signalNeurona[i][j] = sigmoidal(summa);
            }
            else
            {
                if (PARAMETR.function == 1)
                {
                    signalNeurona[i][j] = sigmoidal(summa);
                }
                if (PARAMETR.function == 2)
                {
                    signalNeurona[i][j] = RELU(summa);
                }
            }
            summa = 0.00;
            z = 0;
        }

        z = PARAMETR.layer - i;
        //вычисление выходного слоя
        if (z == 1)
        {
            for (int n = 0; n < PARAMETR.output_Data; n++)
            {
                output[n] = signalNeurona[i][n];
            }
        }
        z = 0;

    }
    //вывод сигналов нейронов на консоль
    /*cout << endl << endl << "Neurons signals" << endl << endl;
    for (int i = 0; i < PARAMETR.layer; i++)
    {
    for (int n = 0; n < PARAMETR.neuron; n++)
    {
    cout << "signalNeurona" << n << "=" << signalNeurona[i][n] << "     ";
    }
    cout << "layer=" << i << endl;
    }
    getchar();*/
    return  0;
}

//вычисление ошибок
double ERROR(vector<vector<double>>& deltaNeurons, vector<double>& outputArray, vector<double>& output,
    vector<vector<vector<double>>>& WEIGHT, vector<vector<double>>& signalNeurona, vector<vector<double>>& summaForSigmoid)
{
    PARAMETRS_ANN PARAMETR;
    double summaDelta = 0.00;
    //ошибки выходного слоя
    for (int n = 0; n < PARAMETR.output_Data; n++)
    {
        //deltaNeurons[PARAMETR.layer - 1][n] = (output[n] - outputArray[n]) * DIF_sigmoid(summaForSigmoid[PARAMETR.layer - 1][n]);
        deltaNeurons[PARAMETR.layer - 1][n] = (-outputArray[n] / output[n] + (1 - outputArray[n]) / (1 - output[n])) * DIF_sigmoid(summaForSigmoid[PARAMETR.layer - 1][n]);
    }

    //ошибки нейронов
    for (int i = PARAMETR.layer - 2; i >= 0; i--)
    {
        for (int n = 0; n < PARAMETR.neuron; n++)
        {
            summaDelta = 0.00;
            for (int k = 0; k < PARAMETR.connect; k++)
            {
                summaDelta += deltaNeurons[i + 1][k] * WEIGHT[i + 1][n][k];
            }

            if (PARAMETR.function == 1)
            {
                deltaNeurons[i][n] = summaDelta * DIF_sigmoid(summaForSigmoid[i][n]);
            }
            if (PARAMETR.function == 2)
            {
                deltaNeurons[i][n] = summaDelta * DifRELU(summaForSigmoid[i][n]);
            }

        }
    }
    //вывод ошибок на консоль
    /*cout << endl << endl << "delta Neurons" << endl << endl;
    for (int i = 0; i < PARAMETR.layer; i++)
    {
    for (int n = 0; n < PARAMETR.neuron; n++)
    {
    cout << "deltaNeuron=" << deltaNeurons[i][n] << "     ";
    }
    cout << "layer=" << i << endl;
    }
    getchar();*/
    return 0;
}

//обратное распространение
double BACK_PROPAGATION(vector<vector<vector<double>>>& WEIGHT, vector<vector<double>>& deltaNeurons, vector<vector<double>>& summaForSigmoid, vector<double>& Input_Array,
    vector<vector<double>>& signalNeurona, vector<vector<double>>& BIAS)
{
    PARAMETRS_ANN PARAM;
    double znachenie = 0.00;
    //вычисление весовых коэффициентов для входного слоя
    for (int i = 0; i < 1; i++)
    {
        for (int n = 0; n < PARAM.neuron; n++)
        {
            for (int k = 0; k < PARAM.connect; k++)
            {
                WEIGHT[i][n][k] = WEIGHT[i][n][k] - (PARAM.alpha * deltaNeurons[i][n] * Input_Array[k]);
            }
        }
    }
    //вычисление весовых коэффициентов для скрытого и выходного слоев
    for (int i = 1; i < PARAM.layer; i++)
    {
        for (int n = 0; n < PARAM.neuron; n++)
        {
            for (int k = 0; k < PARAM.connect; k++)
            {
                WEIGHT[i][n][k] = WEIGHT[i][n][k] - (PARAM.alpha * deltaNeurons[i][n] * signalNeurona[i - 1][k]);
            }
        }
    }
    //вычисление смещений
    for (int i = 0; i < PARAM.layer; i++)
    {
        for (int n = 0; n < PARAM.neuron; n++)
        {
            BIAS[i][n] = BIAS[i][n] - PARAM.alpha * deltaNeurons[i][n];
        }
        //cout << endl;
    }
    // веса после изменений
    /*cout << "change  weight" << endl;
    for (int i = 0; i < layer; i++)
    {
    cout << "layer=" << i << endl;
    for (int j = 0; j < neuron; j++)
    {
    for (int k = 0; k < connect; k++)
    {
    cout << "weightConnect" << k << "=" << WEIGHT[i][j][k] << "   ";
    }
    cout << "neuron=" << j << endl;
    }

    }
    //смещения после изменений
    cout << "change  bias" << endl;
    for (int i = 0; i < layer - 1; i++)
    {
    for (int j = 0; j < neuron; j++)
    {
    cout << "BIAS=" << BIAS[i][j] << "   ";
    }
    cout << endl;
    }
    getchar();*/
    return 0;
}

//запись весовых коэффициентов и смещений в файл
double WeightAndBiasFileOut(vector<vector<vector<double>>>& WEIGHT, vector<vector<double>>& BIAS)
{
    PARAMETRS_ANN PARAMETR;
    ofstream file;
    file.open("koefficientsWeightOut.txt");
    if (file.is_open())
    {
        for (int i = 0; i < PARAMETR.layer; i++)
        {
            for (int n = 0; n < PARAMETR.neuron; n++)
            {
                for (int k = 0; k < PARAMETR.connect; k++)
                {
                    file << WEIGHT[i][n][k] << "   ";
                }
                file << endl;
            }
            //file << endl;
        }
    }
    file.close();
    file.open("koefficientsBiasOut.txt");
    if (file.is_open())
    {
        for (int i = 0; i < PARAMETR.layer; i++)
        {
            for (int n = 0; n < PARAMETR.neuron; n++)
            {
                file << BIAS[i][n] << "   ";
            }
            file << endl;
        }
    }
    file.close();
    return 0;
}


int main()
{
    int numberPoint = 0;
    int koord = 0;
    PARAMETRS_ANN PAR;

    //определяем количество точек в файле
    numberPoint = FILE_number_Point(numberPoint);
    //инициализация входного и выходного вектора
    vector<vector<double>> vector_Input(numberPoint, vector<double>(PAR.input_Data));
    vector<vector<double>> vector_Output(numberPoint, vector<double>(PAR.output_Data));
    //определение входного и выходного вектора
    FILE_INPUT_OUTPUT(numberPoint, vector_Input, vector_Output);
    vector<double> min_Input(PAR.input_Data, 0);
    vector<double> max_Input(PAR.input_Data, 0);
    vector<double> min_Output(PAR.output_Data, 0);
    vector<double> max_Output(PAR.output_Data, 0);
    //массив весов
    vector<vector<vector<double>>> WEIGHT(PAR.layer, vector<vector<double>>(PAR.neuron, vector<double>(PAR.connect, 0)));
    //массив смещений
    vector<vector<double>> BIAS(PAR.layer, vector<double>(PAR.neuron, 0));
    //сигнал каждого нейрона
    vector<vector<double>> signalNeurona(PAR.layer, vector<double>(PAR.neuron, 0));
    //инициализацвесов и смещений
    Initialization_WEIGHT_BIAS(WEIGHT, BIAS);

    vector<double> Input_Array(PAR.neuron);
    vector<vector<double>> vector_Output_ANN(numberPoint, vector<double>(PAR.neuron, 0));
    //выходные значения
    vector<double> output(PAR.output_Data, 0);
    //для проверки выходных значений
    vector<double> output_Array(PAR.output_Data, 0);

    //вектор ошибок
    vector<double> deltaArray(PAR.neuron, 0);
    //vector3d streif(length, vector<vector<double>>(length, vector<double>(length, 0)));

    //массив ошибок для каждого нейрона
    vector<vector<double>> deltaNeurons(PAR.layer, vector<double>(PAR.neuron, 0));

    //массив сумм
    vector<vector<double>> summaForSigmoid(PAR.layer, vector<double>(PAR.neuron, 0));

    //нормируем входной и выходной вектор
    NORMIROVKA(numberPoint, vector_Input, vector_Output, min_Input, max_Input, min_Output, max_Output);
    /*
    for (int i = 0; i < numberPoint; i++)
    {
    for (int j = 0; j < PAR.columns_File / 2; j++)
    {
    cout << "vectorInput" << i << j << "=" << vector_Input[i][j] << "      ";

    }
    cout << endl;
    }
    //getchar();
    for (int i = 0; i < numberPoint; i++)
    {
    for (int j = 0; j < PAR.columns_File / 2; j++)
    {
    cout << "vectorOutput" << i << j << "=" << vector_Output[i][j] << "      ";
    }
    cout << endl;
    }
    //getchar();
    */

    int t = 0;
    int q = 0;
    int min_epoch = 0;
    double Error_AVERAGE = 0.00;
    double Min_Error = 1000.00;
    //прогонка нейросети по эпохам
    for (int f = 1; f <= PAR.epochs; f++)
    {
        //подбираем весовые коэффицинты и смещения прогоняя их по точкам и возвращаем веса и смещения
        //старый вариант
        /*for (int i = 0; i < numberPoint; i++)
        {
            for (int n = 0; n < PAR.input_Data; n++)
            {
                Input_Array[n] = vector_Input[i][n];
                output_Array[n] = vector_Output[i][n];
            }
            //в персептроне производим прямое распространение
            PERCEPTRON(WEIGHT, BIAS, signalNeurona, output, summaForSigmoid, Input_Array);

            //высчитываем ошибки
            ERROR(deltaNeurons, output_Array, output, WEIGHT, signalNeurona, summaForSigmoid);

            //обратное распространение
            BACK_PROPAGATION(WEIGHT, deltaNeurons, summaForSigmoid, Input_Array, signalNeurona, BIAS);
        } */
        //через мини-батчи
        for (int i = PAR.BEGIN_batch; i < PAR.END_batch; i++)
        {
            koord = rand() % numberPoint;

            for (int n = 0; n < PAR.input_Data; n++)
            {
                Input_Array[n] = vector_Input[koord][n];                
                //Input_Array[n] = vector_Input[i][n];                
            }
            for (int n = 0; n < PAR.output_Data; n++)
            {
                output_Array[n] = vector_Output[koord][n];
                //output_Array[n] = vector_Output[i][n];
            }
            //в персептроне производим прямое распространение  
            PERCEPTRON(WEIGHT, BIAS, signalNeurona, output, summaForSigmoid, Input_Array);

            //высчитываем ошибки
            ERROR(deltaNeurons, output_Array, output, WEIGHT, signalNeurona, summaForSigmoid);

            //обратное распространение
            BACK_PROPAGATION(WEIGHT, deltaNeurons, summaForSigmoid, Input_Array, signalNeurona, BIAS);
        }
        //расчет ошибки после каждой эпохи
        for (int d = 0; d < numberPoint; d++)
        {
            for (int y = 0; y < PAR.input_Data; y++)
            {
                Input_Array[y] = vector_Input[d][y];
            }
            //в персептроне производим прямое распространение
            PERCEPTRON(WEIGHT, BIAS, signalNeurona, output, summaForSigmoid, Input_Array);
            for (int h = 0; h < PAR.output_Data; h++)
            {
                vector_Output_ANN[d][h] = output[h];
            }
        }
        for (int c = 0; c < numberPoint; c++)
        {
            for (int r = 0; r < PAR.output_Data; r++)
            {
                Error_AVERAGE += pow((vector_Output[c][r] - vector_Output_ANN[c][r]), 2);
            }
        }
        Error_AVERAGE = Error_AVERAGE / (numberPoint * PAR.output_Data);
        if (Error_AVERAGE < Min_Error)
        {
            Min_Error = Error_AVERAGE;
            min_epoch = f;
        }

        //показ графиков после определенной эпохи
        q = q + 1;
        if (q == PAR.counter)
        {
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_1, PAR.plot_8);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_2, PAR.plot_9);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_3, PAR.plot_10);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_4, PAR.plot_11);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_5, PAR.plot_12);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_6, PAR.plot_13);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_7, PAR.plot_14);
            WeightAndBiasFileOut(WEIGHT, BIAS);
            q = 0;
        }
        //показ графиков при заданной точности 
        if (Error_AVERAGE < PAR.tochnost)
        {
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_1, PAR.plot_8);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_2, PAR.plot_9);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_3, PAR.plot_10);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_4, PAR.plot_11);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_5, PAR.plot_12);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_6, PAR.plot_13);
            GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_7, PAR.plot_14);
            WeightAndBiasFileOut(WEIGHT, BIAS);
        }
        cout << "epochs=" << f << " Error=" << Error_AVERAGE << endl;
        Error_AVERAGE = 0.00;
    }


    cout << "OUTPUT ALL POINT" << endl;
    for (int i = 0; i < numberPoint; i++)
    {
        for (int n = 0; n < PAR.input_Data; n++)
        {
            Input_Array[n] = vector_Input[i][n];
        }
        //в персептроне производим прямое распространение  
        PERCEPTRON(WEIGHT, BIAS, signalNeurona, output, summaForSigmoid, Input_Array);
        for (int n = 0; n < PAR.output_Data; n++)
        {
            vector_Output_ANN[i][n] = output[n];
        }
    }
    WeightAndBiasFileOut(WEIGHT, BIAS);
    GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_1, PAR.plot_8);
    GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_2, PAR.plot_9);
    GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_3, PAR.plot_10);
    GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_4, PAR.plot_11);
    GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_5, PAR.plot_12);
    GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_6, PAR.plot_13);
    GNUPLOT(numberPoint, vector_Output, vector_Output_ANN, PAR.plot_7, PAR.plot_14);
    cout << "Min_epochs=" << min_epoch << " Min_Error=" << Min_Error << endl;
    std::cout << "Hello World!\n";
}// NEURAL_NETWORK.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
