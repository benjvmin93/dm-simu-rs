#[derive(Debug)]
pub enum Plane {
    XY,
    YZ,
    ZX
}

#[derive(Debug)]
pub enum Command {
    N(usize), // N(node)
    M(usize, Plane, f64, Vec<usize>, Vec<usize>, usize),    // M(node, plane, angle, s_domain, t_domain, vop)
    E((usize, usize)),  // E(node1, node2)
    C(usize, usize),    // C(node, cliff_index)
    X(usize, Vec<usize>),  // X(node, domain)
    Z(usize, Vec<usize>),  // Z(node, domain)
    T,  // T
    S(usize, Vec<usize>)   // S(node, domain)
}

#[derive(Debug)]
pub struct Pattern {
    input_nodes: Vec<usize>,
    output_nodes: Vec<usize>,
    n_nodes: usize,
    seq: Vec<Command>,
}

impl Pattern {
    pub fn new(input_nodes: Vec<usize>) -> Self {
        Pattern { 
            input_nodes: input_nodes.iter().map(|e| *e).collect(),
            output_nodes: input_nodes.iter().map(|e| *e).collect(),
            n_nodes: input_nodes.len(),
            seq: Vec::new()
        }
    }

    pub fn add(&mut self, command: Command) {
        if let Command::N(node) = command {
            if self.output_nodes.contains(&node) {
                panic!("Node already prepared!");
            }
            self.n_nodes += 1;
            self.output_nodes.push(node);
        }
        if let Command::M(node, _, _, _, _, _) = command {
            let m_index = self.output_nodes.iter().position(|n| *n == node).unwrap();
            self.output_nodes.remove(m_index);
        }
        self.seq.push(command);
    }

    pub fn print_pattern(&self) {
        println!("{:?}", *self);
    }

    pub fn extend(&mut self, commands: Vec<Command>) {
        self.seq.extend(commands);
    }
}

#[cfg(test)]
mod pattern_tests {
    use super::Pattern;
    use super::Command;

    #[test]
    fn test_init_empty() {
        /*
            Test for initializing empty pattern.
         */
        let _pattern = Pattern::new(Vec::new());
        assert!(_pattern.input_nodes.is_empty());
        assert!(_pattern.output_nodes.is_empty());
        assert!(_pattern.n_nodes == 0);
        assert!(_pattern.seq.is_empty());
    }
    #[test]
    fn test_init_non_empty() {
        /*
            Test for initializing empty pattern.
         */
        let input_nodes: [usize; 5] = [1, 2, 3, 4, 5];
        let _pattern = Pattern::new(input_nodes.iter().map(|n| *n).collect::<Vec<usize>>());
        assert!(_pattern.input_nodes.len() == 5);
        assert!(_pattern.output_nodes.len() == 5);
        assert!(_pattern.n_nodes == 5);
        assert!(_pattern.seq.is_empty());
    }
    #[test]
    fn test_add() {
        /*
            Test for adding five N commands on the input nodes.
         */
        let input_nodes = (1..=5).collect();
        let mut _pattern = Pattern::new(input_nodes);
        _pattern.seq = _pattern.input_nodes.iter().map(|n| Command::N(*n)).collect::<Vec<_>>();
        for elt in &_pattern.seq {
            assert!(matches!(elt, Command::N(_)))
        }

    }
}