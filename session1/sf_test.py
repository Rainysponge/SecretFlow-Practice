import secretflow as sf
sf.shutdown()
sf.init(parties=['alice', 'bob', 'carol'], address='local')  # 建立三个节点,分别是 alice bob carol
alice_device = sf.PYU('alice')
message_from_alice = alice_device(lambda x:x)("Hello World!")

print(message_from_alice)
print(sf.reveal(message_from_alice))  # 解密