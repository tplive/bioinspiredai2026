#[derive(Debug)]
struct Item {
    i: usize,
    p: usize,
    w: usize,
}

fn main() {
    
    // Read data from project1/knapsack/knapPI_12_500_1000_82.csv
    // The data format is I(ndex), p(rofit), w(eight)
    // The knapsack capacity is given at 280785 units
    // We need to find the combination of items where 
    // - the profit is highest, given that 
    // - the total weight of items don't exceed 280785

    let item = Item { 
        i: 1,
        p: 2,
        w: 3
    };

    println!("{:?}", item)
}
