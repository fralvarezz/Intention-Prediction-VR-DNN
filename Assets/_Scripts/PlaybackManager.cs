using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlaybackManager : MonoBehaviour
{
    public string playbackFileName;
    
    public GameObject playerHead;
    public GameObject controller;
    public List<GameObject> items;
    
    public Queue<ReplayData> replayQueue;
    
    public bool isPlaying = false;
    private bool _anyFramesLeft = true;
    
    private float[] frame;

    private static PlaybackManager _instance;
    public static PlaybackManager Instance => _instance;


    private void Awake()
    {
        Application.targetFrameRate = 90;
        
        if (_instance != null && _instance != this)
        {
            Destroy(this.gameObject);
        } else {
            _instance = this;
        }
    }

    void Start()
    {
        replayQueue = CSVParser.ReplayDataQueue(playbackFileName);
        if(replayQueue != null && replayQueue.Count > 0)
        {
            _anyFramesLeft = true;
        }

        frame = new float[24];
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            isPlaying = !isPlaying;
        }
    }

    void LateUpdate()
    {
        if (!isPlaying || !_anyFramesLeft)
        {
            isPlaying = false;
            return;
        }
        
        PlayFrame();
    }

    void PlayFrame()
    {
        if(replayQueue == null || replayQueue.Count == 0)
        {
            _anyFramesLeft = false;
            return;
        }
        
        var data = replayQueue.Dequeue();
        playerHead.transform.position = data.playerHeadPosition;
        playerHead.transform.up = data.playerHeadUp;
        playerHead.transform.forward = data.playerHeadForward;
        controller.transform.position = SetRelativePosition(playerHead.transform, data.relativeControllerPosition);
        controller.transform.up = data.relativeControllerUp;
        controller.transform.forward = data.relativeControllerForward;
        frame = data.GetData();
        Debug.DrawRay(playerHead.transform.position, playerHead.transform.TransformDirection(data.gazeVector.normalized * 10f), Color.red);
        for (int i = 0; i < items.Count; i++)
        {
            if (items[i] != null)
            {
                items[i].GetComponent<Renderer>().material.color = data.objectTag == i ? Color.red : Color.white;
            }
        }
    }
    
    private string VecToStr(Vector3 vec)
    {
        return vec.x.ToString("N4") + ", " + vec.y.ToString("N4") + ", " + vec.z.ToString("N4");
    }
    
    private Vector3 SetRelativePosition(Transform origin, Vector3 position)
    {
        return origin.position + position;
    }
    
    public float[] GetFrame()
    {
        return frame;
    }
}
