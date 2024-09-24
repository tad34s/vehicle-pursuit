using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

public class RegisterDataChannel : MonoBehaviour{
	DataChannel dataChannel;

	public void Awake(){
		dataChannel = new DataChannel();

		SideChannelManager.RegisterSideChannel(dataChannel);
	}

	public void onDestroy(){
		if(Academy.IsInitialized){
			SideChannelManager.UnregisterSideChannel(dataChannel);
		}
	}
}