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
    
    private bool _isPlaying = false;
    private bool _anyFramesLeft = true;

    private void Awake()
    {
        
    }

    void Start()
    {
        replayQueue = CSVParser.ReplayDataQueue(playbackFileName);
        if(replayQueue != null && replayQueue.Count > 0)
        {
            _anyFramesLeft = true;
        }
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            _isPlaying = !_isPlaying;
        }
    }

    void LateUpdate()
    {
        if (!_isPlaying || !_anyFramesLeft)
        {
            _isPlaying = false;
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
        Debug.Log(data.playerHeadPosition);
        Debug.Log(data.relativeControllerPosition);
        playerHead.transform.position = data.playerHeadPosition;
        controller.transform.localPosition = data.relativeControllerPosition;
        Debug.DrawRay(playerHead.transform.position, data.gazeVector.normalized * 10f, Color.red);
        for (int i = 0; i < items.Count; i++)
        {
            if (items[i] != null)
            {
                items[i].GetComponent<Renderer>().material.color = data.objectTag == i ? Color.red : Color.white;
            }
        }

    }
}
