using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using System.IO;
using System.Linq;

public class CSVParser
{
    static string path = "Assets/CSVs/NewData/jonas_data.csv";
    
    public static List<float> minVals = new List<float>()
    {
        -0.8447f,
        0.3382f,
        -0.3764f,
        -2.92f,
        -1.3614f,
        -0.358f,
        -0.9971f,
        -0.6687f,
        -0.9999f,
        -1.0f,
        -1.0f,
        -1.0f,
        -5.15f,
        -0.4183f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        -1.5f
    };

    //TODO: Update maxVals
    public static List<float> maxVals = new List<float>()
    {
        0.5001f, 
        1.0f, 
        0.6271f, 
        0.4839f, 
        0.3688f, 
        1.23f, 
        0.7585f, 
        1.0f, 
        0.2298f,
        0.5054f, 
        0.423f, 
        1.0f,
        0.0f, 
        1.7237f,
        20.0f, 
        9.0f, 
        2204.172f, 
        83815.45f, 
        18.8792f
    };
    
    public static float Normalize(float val, float min, float max)
    {
        var newVal = (val - min) / (max - min);
        //Debug.Log("Converted " + val + " to " + newVal);
        return newVal;
    }
    
    public static List<List<float[,]>> ParseCSV(int keepEvery = 1, bool normalize = false)
    {
        CultureInfo.CurrentCulture = new CultureInfo("da-DK");
        List<float[]> parsedData = new List<float[]>();
        StreamReader reader = new StreamReader(path);
        string line;
        string header = reader.ReadLine();
        int frameCount = 0;
        
        while ((line = reader.ReadLine()) != null)
        {
            if (frameCount % keepEvery != 0)
            {
                frameCount++;
                continue;
            }
            var csvLine = line.Split(';');
            
            csvLine = new []
            {
                csvLine[5], csvLine[6], csvLine[7], csvLine[8], csvLine[9], csvLine[10], csvLine[11], csvLine[12],
                csvLine[13], csvLine[14], csvLine[15], csvLine[16], csvLine[17], csvLine[18], csvLine[19], csvLine[20],
                csvLine[21], csvLine[22], csvLine[23], csvLine[24]
            };

            if(csvLine[csvLine.Length - 1] != "0")
            {
                parsedData.Add(csvLine.Select(float.Parse).ToArray());
                if (normalize)
                {
                    for (int i = 0; i < 19; i++)
                    {
                        parsedData[parsedData.Count - 1][i] = Normalize(parsedData[parsedData.Count - 1][i], minVals[i], maxVals[i]);
                    }
                }
            }
            frameCount++;
        }

        List<List<float[,]>> parsedDataList = new List<List<float[,]>>()
        {
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>(),
            new List<float[,]>()
        };

        int currentClass = 0;
        int initialIndex = 0;
        
        bool lookingForLastIndex = false;
        for (int i = 0; i < parsedData.Count; i++)
        {
            if ((int) parsedData[i][parsedData[i].Length - 1] == currentClass) continue;
            
            if (lookingForLastIndex)
            {
                int previousClass = (int) parsedData[i - 1][parsedData[i - 1].Length - 1];
                float[,] segment = new float[i - initialIndex, parsedData[i - 1].Length];
                    
                for (int j = initialIndex; j < i; j++)
                {
                    for (int k = 0; k < parsedData[j].Length; k++)
                    {
                        segment[j - initialIndex, k] = parsedData[j][k];
                    }
                }
                parsedDataList[previousClass].Add(segment);
            }
            
            currentClass = (int) parsedData[i][parsedData[i].Length - 1];
            initialIndex = i;
            lookingForLastIndex = true;
        }

        return parsedDataList;
    }

    public static Queue<ReplayData> ReplayDataQueue(string filePath)
    {
        CultureInfo cultureInfo = new CultureInfo("da-DK");
        var path = filePath;
        Queue<ReplayData> replayDataQueue = new Queue<ReplayData>();
        
        StreamReader reader = new StreamReader(path);
        string line;
        string header = reader.ReadLine();
        
        while ((line = reader.ReadLine()) != null)
        {
            var csvLine = line.Split(';');

            Vector3 playerHeadPosition = new Vector3(float.Parse(csvLine[2], cultureInfo), float.Parse(csvLine[3], cultureInfo), float.Parse(csvLine[4], cultureInfo));
            Vector3 playerHeadUp = new Vector3(float.Parse(csvLine[5], cultureInfo), float.Parse(csvLine[6], cultureInfo), float.Parse(csvLine[7], cultureInfo));
            Vector3 playerHeadForward = new Vector3(float.Parse(csvLine[8], cultureInfo), float.Parse(csvLine[9], cultureInfo), float.Parse(csvLine[10], cultureInfo));
            
            Vector3 relativeControllerPosition = new Vector3(float.Parse(csvLine[11], cultureInfo), float.Parse(csvLine[12], cultureInfo), float.Parse(csvLine[13], cultureInfo));
            Vector3 relativeControllerUp = new Vector3(float.Parse(csvLine[14], cultureInfo), float.Parse(csvLine[15], cultureInfo), float.Parse(csvLine[16], cultureInfo));
            Vector3 relativeControllerForward = new Vector3(float.Parse(csvLine[17], cultureInfo), float.Parse(csvLine[18], cultureInfo), float.Parse(csvLine[19], cultureInfo));
            
            Vector3 gazeVector = new Vector3(float.Parse(csvLine[20], cultureInfo), float.Parse(csvLine[21], cultureInfo), float.Parse(csvLine[22], cultureInfo));
            int objectTag = int.Parse(csvLine[csvLine.Length - 1]);
            
            ReplayData replayData = new ReplayData(playerHeadPosition, playerHeadUp, playerHeadForward, relativeControllerPosition, relativeControllerUp, relativeControllerForward, gazeVector, objectTag);            
            replayDataQueue.Enqueue(replayData);
        }

        return replayDataQueue;
    }


}
