using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System;
using System.Collections.Generic;

public class DataChannel : SideChannel{
	static private Dictionary<string, int> intParameters = new Dictionary<string, int>();

	static public int getParameter(string key, int defaultValue){
		if(intParameters.ContainsKey(key))
			return intParameters[key];
		return defaultValue;
	}

	public DataChannel(){
		ChannelId = new Guid("9bc23f51-e0e8-450c-b3c5-e4d2032151ec");
	}

	protected override void OnMessageReceived(IncomingMessage msg){
		string receivedString = msg.ReadString();
	
		string[] segments = receivedString.Split('|');
		foreach(string seg in segments){
			Debug.Log(seg);
		}
		switch(segments[0]){
			case "int":
				intParameters[segments[1]] = int.Parse(segments[2]);
				break;
			default:
				Debug.LogError("Didn't recognize data type of message");
				break;
		}
	}
}