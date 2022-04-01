using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR;

public class EyeLogger : MonoBehaviour
{
    public bool logging;
    
    // Stream Writer variables:
    private StreamWriter writer;
    private int frame = 0;
    private DateTime time;
    
    //Player
    public GameObject player;
    private Vector3 playerPosition;
    private Quaternion playerRotation;
    
    //Right Hand
    public GameObject rightHand;
    private Vector3 rightHandPosition;
    private Quaternion rightHandRotation;
    
    //Left Hand
    public GameObject leftHand;
    private Vector3 leftHandPosition;
    private Quaternion leftHandRotation;
    
    //Gaze
    private Vector3 gazeVector;
    private Vector3 gazePoint;
    private string gazeObjectTag;
    
    private static EyeLogger _instance;

    public static EyeLogger Instance { get { return _instance; } }

    private int _logIndex;


    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(this.gameObject);
        } else {
            _instance = this;
        }
    }
    
    void Start()
    {
        _logIndex = PlayerPrefs.GetInt("Index", 0);
        _logIndex++;
        PlayerPrefs.SetInt("Index", _logIndex);
        
        writer = new StreamWriter(GetPath());
        
        writer.WriteLine("Frame;" +
                         "Timestamp;" +
                         "Player Position;" +
                         "Player Rotation;" +
                         "Right Hand Position;" +
                         "Right Hand Rotation;" +
                         "Left Hand Position;" +
                         "Left Hand Rotation;" +
                         "Gaze Vector;" +
                         "Gaze Position;" +
                         "Gaze Object Tag;" + 
                         "Gaze Position Pixel Space"
                         );
        //gaze position in pixel space
        //(dont collect yet) object position
        
    }
    
    private void OnApplicationQuit()
    {
        writer.Close();
    }
    
    void Update()
    {
        UpdateValues();
        
        if (Input.GetKeyDown(KeyCode.Space))
            logging = !logging;

        if(!logging)
            return;
    }

    public void Log(Vector3 gazeVec, Vector3 gazePt, string gazeObj)
    {
        gazeVector = gazeVec;
        gazePoint = gazePt;
        gazeObjectTag = gazeObj;
        
        Debug.Log(gazeObj);
        
        if(!logging)
            return;
        
        writer.WriteLine(frame + ";" +
                         time + " " + time.Millisecond + ";" +
                         playerPosition.ToString("N4") + ";" +
                         playerRotation.ToString("N4") + ";" +
                         rightHandPosition.ToString("N4") + ";" +
                         rightHandRotation.ToString("N4") + ";" +
                         leftHandPosition.ToString("N4") + ";" +
                         leftHandRotation.ToString("N4") + ";" +
                         gazeVector.ToString("N4") + ";" +
                         gazePoint.ToString("N4") + ";" +
                         gazeObjectTag + ";" +
                         Camera.main.WorldToScreenPoint(gazePoint) + ";" //compare this to WorldToScreenVR. Not sure which one works or doesn't.
                         //WorldToScreenVR(Camera.main, gazePoint) + ";"
        );
    }

    private void UpdateValues()
    {
        time = DateTime.Now;
        
        playerPosition = player.transform.position;
        playerRotation = player.transform.rotation;

        rightHandPosition = rightHand.transform.position;
        rightHandRotation = rightHand.transform.rotation;
        
        leftHandPosition = leftHand.transform.position;
        leftHandRotation = leftHand.transform.rotation;
    }
    
    private string GetPath()
    {
#if UNITY_EDITOR
        return Application.dataPath + $"/eyeLog_{_logIndex}.csv";
#endif
    }
    
    /// <summary>
    /// Calculates screen-space position a world space object. Useful for showing something on screen that is not visible in VR.
    /// For example, it can be used to update the position of a marker that highlights the gaze of the player, using eye tracking.
    /// </summary>
    /// <param name="camera">The camera used for VR rendering.</param>
    /// <param name="worldPos">World position of a point.</param>
    /// <returns>Screen position of a point.</returns>
    static Vector2 WorldToScreenVR(Camera camera, Vector3 worldPos)
    {
        Vector3 screenPoint = camera.WorldToViewportPoint(worldPos);
        float w = XRSettings.eyeTextureWidth;
        float h = XRSettings.eyeTextureHeight;
        float ar = w / h;

        screenPoint.x = (screenPoint.x - 0.15f * XRSettings.eyeTextureWidth) / 0.7f;
        screenPoint.y = (screenPoint.y - 0.15f * XRSettings.eyeTextureHeight) / 0.7f;

        return screenPoint;
    }
}
