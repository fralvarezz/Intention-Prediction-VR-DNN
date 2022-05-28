using System.Collections;
using System.Collections.Generic;
using _Scripts;
using UnityEngine;

public class NetworkInference : MonoBehaviour
{
    public enum RunType
    {
        Runtime,
        Replay
    }
    
    SocketInterface socketInterface;

    public int keepEvery;
    public RunType runType;
    private int frameCount = 0;
    
    private GUIStyle currentStyle = null;
    private string STATUS = "";
    private string connected = "CONNECTED";
    private string disconnected = "DISCONNECTED";
    private string debugLogString = "";
    private string debugTimestampString = "";
    
    public Dictionary<int, string> intToTag = new Dictionary<int, string>()
    {
        {0, "None"},
        {-9999, "None"},
        {1, "Poles"},
        {2, "Helmet"},
        {3, "Backpack"},
        {4, "Phone"},
        {5, "Float Ring"},
        {6, "Glasses"},
        {7, "Headphones"},
        {8, "Snowboard"},
        {9, "Bottle"}
    };
    
    void Start()
    {
        socketInterface = gameObject.AddComponent<SocketInterface>();
        
        socketInterface.Connect("localhost", 18500);
    }

    void Update()
    {
        bool connectedCheck = socketInterface.IsConnected();
        
        if (frameCount % keepEvery == 0 && connectedCheck)
        {
            float[] frame = new float[24];
            
            if(runType == RunType.Runtime)
            {
                frame = EyeLogger.Instance.GetFrame();
            }
            else if(runType == RunType.Replay && PlaybackManager.Instance.isPlaying)
            {
                frame = PlaybackManager.Instance.GetFrame();
            }

            socketInterface.SendFrame(frame);
        }
        
        STATUS = connectedCheck ? connected : disconnected;

        frameCount++;
    }
    
    private void OnGUI()
    {
        InitStyles();
        var statusText = STATUS + "\n";
        bool foundFirst = socketInterface.getPredition() != -9999;
        statusText += debugLogString + (foundFirst ? intToTag[socketInterface.getPredition()] : "N/A") + "\n";
        statusText += debugTimestampString + (foundFirst ? socketInterface.getTimestamp().ToString("HH:mm:ss\\Z") : "N/A") + "\n";
        GUI.Label(new Rect(10, 10, 250, 60), statusText, currentStyle);
    }
    
    private void InitStyles()
    {
        if( currentStyle == null )
        {
            currentStyle = new GUIStyle( GUI.skin.box );
            currentStyle.normal.background = MakeTex( 1, 1, new Color( 0f, 0f, 0f, 0.5f ) );
        }
    }
    
    private Texture2D MakeTex( int width, int height, Color col )
    {
        Color[] pix = new Color[width * height];
        for( int i = 0; i < pix.Length; ++i )
        {
            pix[ i ] = col;
        }
        Texture2D result = new Texture2D( width, height );
        result.SetPixels( pix );
        result.Apply();
        return result;
    }
}
