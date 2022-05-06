using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public class ExperimentManager : MonoBehaviour
{
    private static ExperimentManager _instance;

    public static ExperimentManager instance => _instance;

    public List<GameObject> items;
    public float timeBetweenItems;
    private float _timeUntilNextItem;
    private List<GameObject> itemsToHighlight;
    private int _highlightedItemId;
    private bool _anyItemHighlighted;
    public bool random;
    public bool isGuided;
    public int loopCount;

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
        _timeUntilNextItem = timeBetweenItems;
        itemsToHighlight = new List<GameObject>(items);
    }

    void Update()
    {
        if (!isGuided)
            return;
        
        if(_anyItemHighlighted)
            return;
        
        if(loopCount < 0)
            return;

        _timeUntilNextItem -= Time.deltaTime;
        if (_timeUntilNextItem <= 0)
        {
            HighlightItem();
        }
    }
    
    void HighlightItem()
    {
        var index = random ? Random.Range(0, itemsToHighlight.Count - 1) : 0;
        itemsToHighlight[index].GetComponent<Renderer>().material.color = Color.red;
        itemsToHighlight[index].GetComponent<SelectableItem>().isSelected = true;
        _highlightedItemId = index;
        EyeLogger.Instance.correctLabel = itemsToHighlight[index].name;

        _timeUntilNextItem = timeBetweenItems;
        _anyItemHighlighted = true;
    }
    
    public void UnHighlightItem()
    {
        itemsToHighlight[_highlightedItemId].GetComponent<Renderer>().material.color = Color.white;
        itemsToHighlight.RemoveAt(_highlightedItemId);
        EyeLogger.Instance.correctLabel = "";
        _anyItemHighlighted = false;
        
        if(itemsToHighlight.Count == 0 && loopCount > 0)
        {
            itemsToHighlight = new List<GameObject>(items);
            Debug.Log("Restarted the experiment");
            loopCount--;
        }
    }
}
