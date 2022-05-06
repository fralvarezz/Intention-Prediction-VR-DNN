using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using System.IO;
using System.Linq;

public class CSVParser
{
    static string path = "Assets/CSVs/NewData/fer_data.csv";
    
    public static List<List<float[,]>> ParseCSV()
    {
        List<float[]> parsedData = new List<float[]>();
        StreamReader reader = new StreamReader(path);
        string line;
        string header = reader.ReadLine();
        while ((line = reader.ReadLine()) != null)
        {
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
            }
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
