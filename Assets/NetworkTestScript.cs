using System.Collections;
using System.Collections.Generic;
using _Scripts;
using UnityEngine;

public class NetworkTestScript : MonoBehaviour
{
    // Start is called before the first frame update
    private SocketInterface socketInterface;
    void Start()
    {
        socketInterface = gameObject.AddComponent<SocketInterface>();
        
        socketInterface.Connect("localhost", "test message");
        SendFrameNetwork();
    }

    // Update is called once per frame
    void Update()
    {
        //socketInterface.SendString("Test Msg");
    }
    
    public void SendFrameNetwork()
    {
        var data = CSVParser.ParseCSV();
        int cnt = 0;
        foreach (var output in data)
        {
            foreach (var segment in output)
            {
                for (int i = 0; i < segment.GetLength(0); i++)
                {
                    var frame = new float[segment.GetLength(1)];
                    for (int j = 0; j < segment.GetLength(1); j++)
                    {
                        frame[j] = segment[i, j];
                    }
                    socketInterface.SendFrame(frame);
                    cnt++;
                    Debug.Log("Sent " + cnt + " reqs");
                }
            }
        }
    }
    
}
