//UNLICENSE
pragma solidity ^0.8.0;
contract clientManager{
    mapping(address => uint64) public clientId;
    uint64 public nextId = 1;

    function register() public returns (uint64) {
        address[] memory client;

        require(clientId[msg.sender] == 0, "Address already registered");
        clientId[msg.sender] = nextId;
        nextId++;
        return clientId[msg.sender];
    }

    // function pickIds(uint64 count) public view returns (uint64[] memory) {
        
    //     require(count > 0, "Count must be greater than zero");
    //     require(count <= nextId - 1, "Count must be less than or equal to the number of registered addresses");

    //     uint64[] memory ids;
    //     uint64 remaining = count;
    //     uint64 current = 1;

    //     while (remaining > 0) {
    //         uint randomseed = uint(keccak256(abi.encodePacked(block.timestamp, msg.sender)));
    //         uint64 random_number = uint64(randomseed);
    //         if (clientId[random_number%(nextId-1)] != 0) {
    //             ids[count - remaining] = current;
    //             remaining--;
    //         }
    //         current++;
    //     }

    //     return ids;
    // }
}
contract watermarkNegotiation {
    //Define the watermark schema.
    struct watermark{
        uint   sign;
        uint[] key;
    }

    mapping(address => uint) public watermark_mapping;
    uint32 watermark_bit = 64;
    uint32 verification_threshold = 50;

    function generate_watermark() public returns (uint) {
        //check if the address already has a mapping
        uint random_watermark = uint(keccak256(abi.encodePacked(block.timestamp, msg.sender)));
        watermark_mapping[msg.sender] = random_watermark;
        return random_watermark;
    }

    function getwatermark_mapping() public view returns (uint) {
        return watermark_mapping[msg.sender];
    }

    function verifyWatermark(address user, uint64 upload_watermark) public view returns (bool){
        if(watermark_mapping[user] == upload_watermark) return true;
        else return false;
    }

}


