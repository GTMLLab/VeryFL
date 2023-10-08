//UNLICENSE
pragma solidity ^0.8.0;

contract watermarkNegotiation {
    mapping(address => uint64) public watermarkMapping;
    uint32 watermark_bit = 64;
    uint32 verification_threshold = 50;

    function generateWatermark() public {
        //check if the address already has a mapping
        uint randomWatermark = uint(keccak256(abi.encodePacked(block.timestamp, msg.sender)));
        uint64 watermark = uint64(randomWatermark);
        watermarkMapping[msg.sender] = watermark;
    }

    function getwatermarkMapping(address user) public view returns (uint64) {
        return watermarkMapping[user];
    }

    function verifyWatermark(address user, uint64 upload_watermark) public view returns (bool){
        if(watermarkMapping[user] == upload_watermark) return true;
        else return false;
    }

}


