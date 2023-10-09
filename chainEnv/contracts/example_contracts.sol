//UNLICENSE
pragma solidity >=0.4.22;

contract SimpleStorage{
    uint storageData;
    address public minter;
    constructor(){
        minter = msg.sender;
    }
    function set(uint x) public{
        require(msg.sender==minter);
        storageData = x;
    }

    function get(uint x) public view returns (uint){
        return storageData;
    }
}
