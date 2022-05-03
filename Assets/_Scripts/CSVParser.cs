using System.Collections;
using System.Collections.Generic;
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

            //replaces commas with dots in csvLine
            for (int i = 0; i < csvLine.Length; i++)
            {
                csvLine[i] = csvLine[i].Replace(',', '.');
            }
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
            
            //Debug.Log("Class: " + currentClass);
            if (lookingForLastIndex)
            {
                int previousClass = (int) parsedData[i - 1][parsedData[i - 1].Length - 1];
                float[,] segment = new float[i - initialIndex, parsedData[i - 1].Length];
                    
                for (int j = initialIndex; j < i; j++)
                {
                    for (int k = 0; k < parsedData[j].Length; k++)
                    {
                        //Debug.Log($"{initialIndex}, {i}");
                        //Debug.Log(parsedData[j][k]);
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
        
    
}
