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
    public bool shouldLoop;

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
        itemsToHighlight = items;
    }

    void Update()
    {
        if (!isGuided)
            return;
        
        if(_anyItemHighlighted)
            return;

        _timeUntilNextItem -= Time.deltaTime;
        if (_timeUntilNextItem <= 0)
        {
            HighlightItem();
        }
    }
    
    void HighlightItem()
    {
        if (random)
        {
            int randomIndex = Random.Range(0, itemsToHighlight.Count);
            itemsToHighlight[randomIndex].GetComponent<Renderer>().material.color = Color.red;
            itemsToHighlight[randomIndex].GetComponent<SelectableItem>().isSelected = true;
            _highlightedItemId = randomIndex;
            EyeLogger.Instance.objectInteractedWith = itemsToHighlight[randomIndex].name;
        }
        else
        {
            itemsToHighlight[0].GetComponent<Renderer>().material.color = Color.red;
            itemsToHighlight[0].GetComponent<SelectableItem>().isSelected = true;
            _highlightedItemId = 0;
            EyeLogger.Instance.objectInteractedWith = itemsToHighlight[0].name;
        }

        _timeUntilNextItem = timeBetweenItems;
        _anyItemHighlighted = true;
    }
    
    public void UnHighlightItem()
    {
        itemsToHighlight[_highlightedItemId].GetComponent<Renderer>().material.color = Color.white;
        itemsToHighlight.RemoveAt(_highlightedItemId);
        EyeLogger.Instance.objectInteractedWith = "";
        _anyItemHighlighted = false;
        
        if(itemsToHighlight.Count == 0 && shouldLoop)
        {
            itemsToHighlight = items;
        }
    }
}
