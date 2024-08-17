from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
	SideChannel,
	IncomingMessage,
	OutgoingMessage
)
import uuid

class DataChannel(SideChannel):
	def __init__(self) -> None:
		super().__init__(uuid.UUID('9bc23f51-e0e8-450c-b3c5-e4d2032151ec'))

	def on_message_received(self, msg: IncomingMessage) -> None:
		print(msg.read_string())

 
	def set_int_parameter(self, key: str, val: int) -> None:
		msg = OutgoingMessage()
		msg.write_string(f'int|{key}|{val}')
		super().queue_message_to_send(msg)